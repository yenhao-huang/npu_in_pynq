# Recreate and optionally implement the PYNQ-Z1 NPU matrix overlay.
# Usage: vivado -mode batch -source build_overlay.tcl -tclargs
#        ?--array-size 2|8? ?--elaborate-only? ?--allow-dirty?

set script_dir [file normalize [file dirname [info script]]]
set repo_root [file normalize [file join $script_dir .. .. .. ..]]
set array_size 2
set elaborate_only 0
set allow_dirty 0
for {set argument_index 0} {$argument_index < [llength $argv]} {incr argument_index} {
    set argument [lindex $argv $argument_index]
    switch -- $argument {
        --array-size {
            incr argument_index
            if {$argument_index >= [llength $argv]} {
                error "--array-size requires 2 or 8"
            }
            set array_size [lindex $argv $argument_index]
            if {$array_size ni {2 8}} {
                error "unsupported array size '$array_size': expected 2 or 8"
            }
        }
        --elaborate-only { set elaborate_only 1 }
        --allow-dirty { set allow_dirty 1 }
        default { error "unknown argument '$argument'" }
    }
}

set build_name npu_matrix
if {$array_size != 2} {
    set build_name "npu_matrix_${array_size}x${array_size}"
}
set build_root [file normalize [file join $repo_root build vivado $build_name]]
set project_dir [file join $build_root project]
set report_dir [file join $build_root reports]
set artifact_dir [file join $build_root artifacts]
set jobs 4

if {!$elaborate_only && !$allow_dirty} {
    if {[catch {exec git -C $repo_root status --porcelain --untracked-files=normal} source_status]} {
        error "cannot verify source tree cleanliness: $source_status"
    }
    if {[string trim $source_status] ne ""} {
        error "refusing production artifact build from a dirty source tree"
    }
}

file delete -force $build_root
file mkdir $project_dir
file mkdir $report_dir
file mkdir $artifact_dir

create_project npu_matrix $project_dir -part xc7z020clg400-1 -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

set rtl_files [list \
    [file join $repo_root src hw rtl systolic_array npu_pe.sv] \
    [file join $repo_root src hw rtl systolic_array npu_systolic_array.sv] \
    [file join $repo_root src hw rtl npu_matrix npu_axi_lite_regs.sv] \
    [file join $repo_root src hw rtl npu_matrix npu_matrix_controller.sv] \
    [file join $repo_root src hw rtl npu_matrix npu_matrix_accelerator.sv]]
foreach rtl_file $rtl_files {
    if {![file isfile $rtl_file]} {
        error "missing source-controlled RTL: $rtl_file"
    }
}
add_files -norecurse $rtl_files
set_property file_type {SystemVerilog} [get_files $rtl_files]
update_compile_order -fileset sources_1

create_bd_design npu_matrix
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* ps7]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000}] $ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
    $ps7

set reset0 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* reset0]
set reset_inverter [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* reset_inverter]
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] $reset_inverter
set control_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* control_ic]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $control_ic
set memory_ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* memory_ic]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $memory_ic
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axis_mm2s_tdata_width {8} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
    CONFIG.c_sg_length_width {23}] $dma
set accelerator [create_bd_cell -type module -reference npu_matrix_accelerator npu_matrix_accelerator_0]
set_property -dict [list \
    CONFIG.ROWS $array_size \
    CONFIG.COLUMNS $array_size \
    CONFIG.MAX_K {256}] $accelerator
set irq_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:* irq_concat]
set_property CONFIG.NUM_PORTS {3} $irq_concat

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins control_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_ic/M00_AXI] [get_bd_intf_pins npu_matrix_accelerator_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins control_ic/M01_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins memory_ic/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins memory_ic/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins memory_ic/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins npu_matrix_accelerator_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins npu_matrix_accelerator_0/m_axis] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
    [get_bd_pins reset0/slowest_sync_clk] \
    [get_bd_pins control_ic/ACLK] [get_bd_pins control_ic/S00_ACLK] \
    [get_bd_pins control_ic/M00_ACLK] [get_bd_pins control_ic/M01_ACLK] \
    [get_bd_pins memory_ic/ACLK] [get_bd_pins memory_ic/S00_ACLK] \
    [get_bd_pins memory_ic/S01_ACLK] [get_bd_pins memory_ic/M00_ACLK] \
    [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_HP0_ACLK] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins npu_matrix_accelerator_0/s_axi_aclk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins reset_inverter/Op1]
connect_bd_net [get_bd_pins reset_inverter/Res] [get_bd_pins reset0/ext_reset_in]
connect_bd_net [get_bd_pins reset0/interconnect_aresetn] \
    [get_bd_pins control_ic/ARESETN] [get_bd_pins control_ic/S00_ARESETN] \
    [get_bd_pins control_ic/M00_ARESETN] [get_bd_pins control_ic/M01_ARESETN] \
    [get_bd_pins memory_ic/ARESETN] [get_bd_pins memory_ic/S00_ARESETN] \
    [get_bd_pins memory_ic/S01_ARESETN] [get_bd_pins memory_ic/M00_ARESETN]
connect_bd_net [get_bd_pins reset0/peripheral_aresetn] \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins npu_matrix_accelerator_0/s_axi_aresetn]
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins irq_concat/In0]
connect_bd_net [get_bd_pins axi_dma_0/s2mm_introut] [get_bd_pins irq_concat/In1]
connect_bd_net [get_bd_pins npu_matrix_accelerator_0/irq] [get_bd_pins irq_concat/In2]
connect_bd_net [get_bd_pins irq_concat/dout] [get_bd_pins ps7/IRQ_F2P]

create_bd_addr_seg -range 0x00010000 -offset 0x43C00000 \
    [get_bd_addr_spaces ps7/Data] \
    [get_bd_addr_segs npu_matrix_accelerator_0/s_axi/reg0] SEG_npu_matrix_accelerator
create_bd_addr_seg -range 0x00010000 -offset 0x40400000 \
    [get_bd_addr_spaces ps7/Data] \
    [get_bd_addr_segs axi_dma_0/S_AXI_LITE/Reg] SEG_axi_dma
assign_bd_address

validate_bd_design
save_bd_design
set bd_file [get_files [file join $project_dir npu_matrix.srcs sources_1 bd npu_matrix npu_matrix.bd]]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set wrapper_file [file join $project_dir npu_matrix.gen sources_1 bd npu_matrix hdl npu_matrix_wrapper.v]
add_files -norecurse $wrapper_file
set_property top npu_matrix_wrapper [current_fileset]
update_compile_order -fileset sources_1

set address_file [open [file join $report_dir address_map.txt] w]
puts $address_file "accelerator=0x43C00000 range=0x00010000"
puts $address_file "dma=0x40400000 range=0x00010000"
foreach segment [get_bd_addr_segs] {
    puts $address_file "[get_property NAME $segment] offset=[get_property OFFSET $segment] range=[get_property RANGE $segment]"
}
close $address_file

if {$elaborate_only} {
    puts "PASS: npu_matrix ${array_size}x${array_size} Vivado block-design elaboration"
    close_project
    exit 0
}

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "synthesis did not complete"
}
open_run synth_1
report_utilization -file [file join $report_dir utilization_synth.rpt]
close_design

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "implementation or write_bitstream did not complete"
}
open_run impl_1
report_utilization -file [file join $report_dir utilization_impl.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -max_paths 10 -file [file join $report_dir timing_summary_routed.rpt]
report_drc -file [file join $report_dir drc_routed.rpt]
set route_report [file join $report_dir route_status.rpt]
report_route_status -file $route_report

set route_handle [open $route_report r]
set route_text [read $route_handle]
close $route_handle
if {![regexp {# of routable nets\.*[[:space:]]*:[[:space:]]*([0-9]+)} \
      $route_text -> routable_nets] ||
    ![regexp {# of fully routed nets\.*[[:space:]]*:[[:space:]]*([0-9]+)} \
      $route_text -> fully_routed_nets] ||
    ![regexp {# of nets with routing errors\.*[[:space:]]*:[[:space:]]*([0-9]+)} \
      $route_text -> routing_errors]} {
    error "route-status report format was not recognized"
}
if {$routable_nets != $fully_routed_nets || $routing_errors != 0} {
    error "route-status gate failed: routable=$routable_nets fully_routed=$fully_routed_nets errors=$routing_errors"
}

set worst_path [get_timing_paths -setup -max_paths 1]
if {[llength $worst_path] == 0} {
    error "no routed setup timing path was found"
}
set wns [get_property SLACK $worst_path]
set failing_paths [get_timing_paths -setup -slack_lesser_than 0 -max_paths 100000]
set drc_errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
if {$wns < 0.0 || [llength $failing_paths] != 0} {
    error "routed timing failed: WNS=$wns failing_setup_paths=[llength $failing_paths]"
}
if {[llength $drc_errors] != 0} {
    error "routed DRC has [llength $drc_errors] error violations"
}

set bit_source [file join $project_dir npu_matrix.runs impl_1 npu_matrix_wrapper.bit]
set hwh_source [file join $project_dir npu_matrix.gen sources_1 bd npu_matrix hw_handoff npu_matrix.hwh]
if {![file isfile $bit_source] || ![file isfile $hwh_source]} {
    error "same-build BIT or HWH artifact is missing"
}
if {$allow_dirty} {
    puts "PASS: npu_matrix dirty exploratory implementation; artifacts not published"
    close_project
    exit 0
}
file copy -force $bit_source [file join $artifact_dir npu_matrix.bit]
file copy -force $hwh_source [file join $artifact_dir npu_matrix.hwh]

set source_commit unknown
catch {set source_commit [string trim [exec git -C $repo_root rev-parse HEAD]]}
set verifier [file join $repo_root src runtime verify_overlay.py]
if {[catch {exec python $verifier --write-manifest \
    --source-commit $source_commit --vivado-version [version -short] \
    --array-size $array_size $artifact_dir} verify_output]} {
    error "overlay provenance verification failed: $verify_output"
}
puts $verify_output

set evidence [open [file join $report_dir build_evidence.txt] w]
puts $evidence "vivado=[version -short]"
puts $evidence "part=xc7z020clg400-1"
puts $evidence "rows=$array_size"
puts $evidence "columns=$array_size"
puts $evidence "wns=$wns"
puts $evidence "setup_failing_paths=[llength $failing_paths]"
puts $evidence "drc_errors=[llength $drc_errors]"
puts $evidence "bit=[file join $artifact_dir npu_matrix.bit]"
puts $evidence "hwh=[file join $artifact_dir npu_matrix.hwh]"
puts $evidence "source_commit=$source_commit"
close $evidence
puts "PASS: npu_matrix ${array_size}x${array_size} Vivado implementation and bitstream"
close_project
exit 0
