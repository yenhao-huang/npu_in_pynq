`timescale 1ns/1ps

module npu_systolic_array #(
    parameter integer ROWS = 2,
    parameter integer COLUMNS = 2,
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    // Leave headroom on the Zynq-7020 (220 DSPs); excess PEs use LUT
    // multipliers with identical registers, valid timing, and arithmetic.
    parameter integer DSP_BUDGET = 192
) (
    input  logic                                      clk,
    input  logic                                      rst_n,
    input  logic                                      clear,
    input  logic                                      enable,
    input  logic signed [ROWS*DATA_WIDTH-1:0]         a_in,
    input  logic [ROWS-1:0]                           a_valid_in,
    input  logic signed [COLUMNS*DATA_WIDTH-1:0]      b_in,
    input  logic [COLUMNS-1:0]                        b_valid_in,
    output wire signed [ROWS*COLUMNS*ACC_WIDTH-1:0]   accumulators
);
    wire signed [DATA_WIDTH-1:0] a_data [0:ROWS-1][0:COLUMNS];
    wire                         a_valid [0:ROWS-1][0:COLUMNS];
    wire signed [DATA_WIDTH-1:0] b_data [0:ROWS][0:COLUMNS-1];
    wire                         b_valid [0:ROWS][0:COLUMNS-1];
    wire signed [ACC_WIDTH-1:0]  accumulator_data [0:ROWS-1][0:COLUMNS-1];

    initial begin
        if (ROWS <= 0) begin
            $fatal(1, "npu_systolic_array requires ROWS > 0");
        end
        if (COLUMNS <= 0) begin
            $fatal(1, "npu_systolic_array requires COLUMNS > 0");
        end
        if (DATA_WIDTH != 8) begin
            $fatal(1, "npu_systolic_array ABI v1 requires DATA_WIDTH=8");
        end
        if (ACC_WIDTH != 32) begin
            $fatal(1, "npu_systolic_array ABI v1 requires ACC_WIDTH=32");
        end
    end

    genvar row;
    genvar column;
    generate
        for (row = 0; row < ROWS; row = row + 1) begin : gen_a_edges
            assign a_data[row][0] = a_in[row*DATA_WIDTH +: DATA_WIDTH];
            assign a_valid[row][0] = a_valid_in[row];
        end

        for (column = 0; column < COLUMNS; column = column + 1) begin : gen_b_edges
            assign b_data[0][column] = b_in[column*DATA_WIDTH +: DATA_WIDTH];
            assign b_valid[0][column] = b_valid_in[column];
        end

        for (row = 0; row < ROWS; row = row + 1) begin : gen_rows
            for (column = 0; column < COLUMNS; column = column + 1) begin : gen_columns
                npu_pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH),
                    .DSP_STYLE((row*COLUMNS+column < DSP_BUDGET) ? "yes" : "no")
                ) pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .clear(clear),
                    .enable(enable),
                    .a_in(a_data[row][column]),
                    .b_in(b_data[row][column]),
                    .a_valid_in(a_valid[row][column]),
                    .b_valid_in(b_valid[row][column]),
                    .a_out(a_data[row][column+1]),
                    .b_out(b_data[row+1][column]),
                    .a_valid_out(a_valid[row][column+1]),
                    .b_valid_out(b_valid[row+1][column]),
                    .accumulator(accumulator_data[row][column])
                );

                assign accumulators[
                    (row*COLUMNS+column)*ACC_WIDTH +: ACC_WIDTH
                ] = accumulator_data[row][column];
            end
        end
    endgenerate
endmodule
