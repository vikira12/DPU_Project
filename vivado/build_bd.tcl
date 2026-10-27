# Zybo Z7-20 DPU 블록 디자인 생성 → 합성·구현·비트스트림 → xsa 내보내기
#
#   vivado -mode batch -source vivado/build_bd.tcl                  (전체)
#   vivado -mode batch -source vivado/build_bd.tcl -tclargs bd      (프로젝트·BD만 생성)
#
# 먼저 HLS IP를 패키징해 둘 것: vitis/dpu_hls/dpu_top/hls/impl/ip
# 결과: vivado/dpu_vivado/ (프로젝트), vivado/dpu_bd_wrapper.xsa

#   Vivado Tcl 콘솔에서: source D:/project/DPU/vivado/build_bd.tcl  (경로는 스크립트 위치 기준이라 현재 디렉터리 무관)
set mode    [expr {[info exists argv] && [llength $argv] > 0 ? [lindex $argv 0] : "all"}]
set here    [file normalize [file dirname [info script]]]
set root    [file normalize "$here/.."]
set ip_repo [file normalize "$root/vitis/dpu_hls/dpu_top/hls/impl/ip"]
set proj    dpu_vivado
set bd      dpu_bd

if {![file exists "$ip_repo/component.xml"]} {
    error "HLS IP가 없음: $ip_repo (HLS 합성·패키징 먼저)"
}

create_project $proj "$here/$proj" -part xc7z020clg400-1 -force
set_property board_part digilentinc.com:zybo-z7-20:part0:1.2 [current_project]
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

create_bd_design $bd

# ---- PS7: 보드 프리셋 + FCLK0 100MHz, GP0(제어), HP0 64비트(DDR 접근) ----
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} $ps
set_property -dict [list \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
] $ps

# ---- DPU ----
set dpu [create_bd_cell -type ip -vlnv xilinx.com:hls:dpu_top:1.0 dpu_0]

# ---- 리셋, 인터커넥트 ----
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M]
set smc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $smc_ctrl
set smc_mem [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_mem]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $smc_mem

# 클록·리셋
set clk  [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net $clk \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK] \
    [get_bd_pins dpu_0/ap_clk] \
    [get_bd_pins rst_ps7_0_100M/slowest_sync_clk] \
    [get_bd_pins axi_smc/aclk] \
    [get_bd_pins axi_smc_mem/aclk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_0_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
    [get_bd_pins dpu_0/ap_rst_n] \
    [get_bd_pins axi_smc/aresetn] \
    [get_bd_pins axi_smc_mem/aresetn]

# AXI
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] [get_bd_intf_pins dpu_0/s_axi_CTRL]
connect_bd_intf_net [get_bd_intf_pins dpu_0/m_axi_gmem0] [get_bd_intf_pins axi_smc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dpu_0/m_axi_gmem1] [get_bd_intf_pins axi_smc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_mem/M00_AXI] [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# 인터럽트 (폴링으로도 동작하지만 연결해 둠)
set_property CONFIG.PCW_USE_FABRIC_INTERRUPT {1} $ps
set_property CONFIG.PCW_IRQ_F2P_INTR {1} $ps
connect_bd_net [get_bd_pins dpu_0/interrupt] [get_bd_pins processing_system7_0/IRQ_F2P]

# 주소: 제어 레지스터 0x40000000, DPU 마스터는 DDR 전체
assign_bd_address -offset 0x40000000 -range 64K [get_bd_addr_segs dpu_0/s_axi_CTRL/Reg]
assign_bd_address

validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files $bd.bd] -top]
add_files -norecurse $wrapper
set_property top ${bd}_wrapper [current_fileset]
update_compile_order -fileset sources_1

if {$mode eq "bd"} { return }

# ---- 합성·구현·비트스트림 ----
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "구현 실패: [get_property STATUS [get_runs impl_1]]"
}

open_run impl_1
report_timing_summary -file "$here/timing_summary.rpt"
report_utilization    -file "$here/utilization.rpt"

write_hw_platform -fixed -include_bit -force -file "$here/${bd}_wrapper.xsa"
puts "xsa: $here/${bd}_wrapper.xsa"
