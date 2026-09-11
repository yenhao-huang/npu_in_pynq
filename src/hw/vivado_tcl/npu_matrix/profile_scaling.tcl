# Isolated controller implementation: no PS/DMA or board-performance claim.
# vivado -mode batch -source .../profile_scaling.tcl -tclargs ROOT OUTPUT SIZE
if {[llength $argv] != 3} { error "expected ROOT OUTPUT SIZE" }
lassign $argv source_root output_root size
set source_root [file normalize $source_root]
set output_root [file normalize $output_root]
if {$size ni {4 8 16}} { error "expected size 4, 8, or 16" }
file mkdir $output_root
cd $output_root
set_param general.maxThreads 4
create_project -in_memory -part xc7z020clg400-1
foreach path {systolic_array/npu_pe.sv systolic_array/npu_systolic_array.sv npu_matrix/npu_matrix_controller.sv} {
    read_verilog -sv [file join $source_root src hw rtl $path]
}
synth_design -top npu_matrix_controller -mode out_of_context -part xc7z020clg400-1 \
    -generic ROWS=$size -generic COLUMNS=$size -generic MAX_K=256
create_clock -period 10.000 [get_ports clk]
report_utilization -file [file join $output_root synth_utilization.rpt]
set utilization_file [open [file join $output_root synth_utilization.rpt] r]
set utilization_text [read $utilization_file]
close $utilization_file
if {![regexp {\| Slice LUTs\*?\s+\|\s*([0-9]+)} $utilization_text unused lut_count]} {
    error "cannot read LUT count from synthesis utilization report"
}
if {$lut_count > 53200} {
    set status_file [open [file join $output_root status.txt] w]
    puts $status_file "UNFIT_LUTS: $lut_count exceeds xc7z020 capacity 53200"
    close $status_file
    close_project
    return
}
report_timing_summary -file [file join $output_root synth_timing.rpt]
write_checkpoint -force [file join $output_root synth.dcp]
set dsp_count [llength [get_cells -hier -filter {REF_NAME =~ DSP48*}]]
if {$dsp_count > 220} {
    set status_file [open [file join $output_root status.txt] w]
    puts $status_file "UNFIT: $dsp_count DSP48 cells exceeds xc7z020 capacity 220"
    close $status_file
} else {
    opt_design
    place_design
    phys_opt_design
    route_design
    report_utilization -file [file join $output_root routed_utilization.rpt]
    report_timing_summary -file [file join $output_root routed_timing.rpt]
    report_timing -max_paths 10 -file [file join $output_root critical_paths.rpt]
    write_checkpoint -force [file join $output_root routed.dcp]
    set status_file [open [file join $output_root status.txt] w]
    puts $status_file "ROUTED"
    close $status_file
}
close_project
