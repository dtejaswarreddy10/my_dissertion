module pe_int8 #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  logic                    clk,
    input  logic                    rst,
    input  logic                    valid_in,
    input  logic                    acc_clear,
    input  logic [DATA_WIDTH-1:0]   a_in,
    input  logic [DATA_WIDTH-1:0]   b_in,
    output logic [DATA_WIDTH-1:0]   a_out,
    output logic [DATA_WIDTH-1:0]   b_out,
    output logic                    valid_out,
    output logic [ACC_WIDTH-1:0]    acc_result
);

    logic signed [ACC_WIDTH-1:0] product;
    assign product = $signed(a_in) * $signed(b_in);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            a_out      <= '0;
            b_out      <= '0;
            valid_out  <= 1'b0;
            acc_result <= '0;
        end else begin
            a_out     <= a_in;
            b_out     <= b_in;
            valid_out <= valid_in;
            if (acc_clear)
                acc_result <= '0;
            else if (valid_in)
                acc_result <= acc_result + product;
        end
    end

endmodule
