`timescale 1ns/1ps

// Cycle accounting and independent signed matrix reference, including stalls
// and changing partial-tile shapes without clearing the operand memories.
module tb_npu_matrix_scaling;
    parameter integer SIZE = 8;
    parameter integer EXPECT_OVERLAP = 1;
    logic clk = 0, rst_n = 0, start_pulse = 0, soft_reset_pulse = 0;
    logic [15:0] cfg_m = 0, cfg_n = 0, cfg_k = 0;
    logic [31:0] cfg_a_stride = 0, cfg_b_stride = 0, cfg_c_stride = 0;
    logic [31:0] cfg_timeout_cycles = 100000;
    logic [7:0] s_axis_tdata = 0, error_code;
    logic s_axis_tvalid = 0, s_axis_tready, s_axis_tlast = 0;
    logic [31:0] m_axis_tdata;
    logic m_axis_tvalid, m_axis_tready = 0, m_axis_tlast;
    logic status_busy, status_done, status_error;
    logic [63:0] cycles;
    integer load_cycles, compute_cycles, output_cycles;
    integer signed av [0:SIZE*256-1], bv [0:256*SIZE-1];
    npu_matrix_controller #(.ROWS(SIZE), .COLUMNS(SIZE), .MAX_K(256)) dut (.*);
    always #5 clk = ~clk;
    always @(posedge clk) begin
        if (status_busy) begin
            if (s_axis_tready) load_cycles = load_cycles + 1;
            else if (m_axis_tvalid) output_cycles = output_cycles + 1;
            else compute_cycles = compute_cycles + 1;
        end
    end

    task automatic run_case(input integer m, n, k, stalls);
        integer i, j, q, out_index, tick;
        integer signed expected;
        logic [31:0] held_data;
        logic held_last, held_valid;
        begin
            @(negedge clk);
            cfg_m = m; cfg_n = n; cfg_k = k;
            cfg_a_stride = k; cfg_b_stride = n; cfg_c_stride = 4*n;
            load_cycles = 0; compute_cycles = 0; output_cycles = 0;
            for (i=0; i<m*k; i=i+1) av[i] = ((i*73 + k*19) % 256)-128;
            for (i=0; i<k*n; i=i+1) bv[i] = ((i*31 + n*7) % 256)-128;
            start_pulse = 1;
            @(negedge clk); start_pulse = 0;
            for (i=0; i<m*k+k*n; i=i+1) begin
                if (stalls && (i % 7 == 0)) begin
                    s_axis_tvalid = 0;
                    @(negedge clk);
                end
                s_axis_tvalid = 1;
                s_axis_tdata = (i < m*k) ? av[i][7:0] : bv[i-m*k][7:0];
                s_axis_tlast = (i == m*k-1 || i == m*k+k*n-1);
                if (!s_axis_tready) $fatal(1, "FAIL input not ready");
                @(negedge clk);
            end
            s_axis_tvalid = 0; s_axis_tlast = 0;
            out_index = 0; tick = 0; held_valid = 0;
            while (!status_done && !status_error && tick < 100000) begin
                m_axis_tready = !stalls || (tick % 5 >= 2);
                @(posedge clk);
                if (held_valid && (!m_axis_tvalid || m_axis_tdata !== held_data ||
                                   m_axis_tlast !== held_last))
                    $fatal(1, "FAIL output changed under backpressure");
                held_valid = m_axis_tvalid && !m_axis_tready;
                held_data = m_axis_tdata; held_last = m_axis_tlast;
                if (m_axis_tvalid && m_axis_tready) begin
                    i = out_index/n; j = out_index%n; expected = 0;
                    for (q=0; q<k; q=q+1) expected = expected + av[i*k+q]*bv[q*n+j];
                    if ($signed(m_axis_tdata) !== expected ||
                        m_axis_tlast !== (out_index == m*n-1))
                        $fatal(1, "FAIL SIZE=%0d m=%0d n=%0d k=%0d index=%0d got=%0d expected=%0d",
                               SIZE,m,n,k,out_index,$signed(m_axis_tdata),expected);
                    out_index = out_index + 1;
                end
                @(negedge clk); tick = tick + 1;
            end
            m_axis_tready = 0;
            if (!status_done || status_error || out_index != m*n)
                $fatal(1, "FAIL incomplete transaction");
            if (cycles != load_cycles+compute_cycles+output_cycles)
                $fatal(1, "FAIL cycle accounting");
            if (!stalls && cycles != m*k+k*n+m*n+k+m+n+1 -
                (EXPECT_OVERLAP ? k-1 : 0))
                $fatal(1, "FAIL no-stall latency regression");
            $display("METRIC size=%0d m=%0d n=%0d k=%0d stalls=%0d cycles=%0d load=%0d compute=%0d output=%0d macs=%0d",
                     SIZE,m,n,k,stalls,cycles,load_cycles,compute_cycles,output_cycles,m*n*k);
        end
    endtask

    // Reset or malformed B after some compute steps have already overlapped
    // loading. The following valid job must not inherit partial products.
    task automatic abort_during_b(input integer malformed);
        integer i;
        begin
            @(negedge clk);
            cfg_m = SIZE; cfg_n = SIZE; cfg_k = 8;
            cfg_a_stride = 8; cfg_b_stride = SIZE; cfg_c_stride = 4*SIZE;
            start_pulse = 1;
            @(negedge clk); start_pulse = 0;
            for (i=0; i<SIZE*8+SIZE*3; i=i+1) begin
                s_axis_tvalid = 1; s_axis_tdata = 127;
                s_axis_tlast = (i == SIZE*8-1) ||
                    (malformed && i == SIZE*8+SIZE*3-1);
                @(negedge clk);
            end
            s_axis_tvalid = 0; s_axis_tlast = 0;
            if (malformed && (!status_error || error_code != 4 || status_busy))
                $fatal(1, "FAIL malformed B accepted after overlapped compute");
            soft_reset_pulse = 1;
            @(negedge clk); soft_reset_pulse = 0;
            if (status_busy || status_done || status_error || cycles != 0)
                $fatal(1, "FAIL soft reset during B");
        end
    endtask

    initial begin
        repeat (3) @(negedge clk); rst_n = 1;
        run_case(SIZE,SIZE,1,0);
        run_case(SIZE,SIZE,32,0);
        run_case(SIZE,SIZE,256,0);
        run_case(SIZE-1,SIZE-2,17,1);
        run_case(1,1,256,1);
        run_case(SIZE,SIZE,1,1);
        abort_during_b(0);
        run_case(SIZE-1,1,32,1);
        abort_during_b(1);
        run_case(1,SIZE-1,32,1);
        $display("PASS tb_npu_matrix_scaling SIZE=%0d", SIZE);
        $finish;
    end
    initial begin
        #2000000; $fatal(1, "FAIL watchdog");
    end
endmodule
