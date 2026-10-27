"""Vivado xsa → Vitis 플랫폼 → 보드 검증 앱(dpu_app) 생성·빌드

사용 (DPU 저장소 루트에서): vitis -s vitis/build_app.py
  - 처음 실행: vitis/ 작업 공간에 dpu_platform, dpu_app 컴포넌트를 만듦
  - 다시 실행: 플랫폼 하드웨어를 새 xsa로 갱신하고 다시 빌드
  Vitis IDE가 이 작업 공간을 열고 있으면 실패하므로 IDE를 닫고 실행
앱은 sw/src의 소스와 generated/ 헤더를 직접 빌드함 (UserConfig.cmake를 이 스크립트가 고쳐 씀)
결과: vitis/dpu_app/build/dpu_app.elf
"""
import os
import re
import shutil
import vitis

WS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WS)
XSA = os.path.join(ROOT, "vivado", "dpu_bd_wrapper.xsa")
PLATFORM = "dpu_platform"
APP = "dpu_app"
DOMAIN = "standalone_ps7_cortexa9_0"
SRC = '"${CMAKE_SOURCE_DIR}/../../../sw/src/'
SOURCES = ["dpu_test.c", "dpu_net.c"]


def set_user_config():
    """UserConfig.cmake의 USER_COMPILE_SOURCES·USER_INCLUDE_DIRECTORIES·최적화 수준 지정"""
    p = os.path.join(WS, APP, "src", "UserConfig.cmake")
    raw = open(p, "rb").read().decode("utf8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    s = re.sub(r"set\(USER_COMPILE_SOURCES\n.*?\)", "set(USER_COMPILE_SOURCES\n" +
               "\n".join(SRC + f + '"' for f in SOURCES) + "\n)", s, count=1, flags=re.S)
    s = re.sub(r"set\(USER_INCLUDE_DIRECTORIES\n.*?\)",
               'set(USER_INCLUDE_DIRECTORIES\n"${CMAKE_SOURCE_DIR}/../../../sw/src"\n)', s, count=1, flags=re.S)
    s = re.sub(r'set\(USER_COMPILE_OPTIMIZATION_LEVEL [^)]*\)', 'set(USER_COMPILE_OPTIMIZATION_LEVEL "-O2")', s)
    s = re.sub(r"set\(USER_LINK_LIBRARIES\n.*?\)", 'set(USER_LINK_LIBRARIES\n"m"\n)', s, count=1, flags=re.S)
    if crlf:
        s = s.replace("\n", "\r\n")
    open(p, "wb").write(s.encode("utf8"))



if not os.path.exists(XSA):
    raise SystemExit(f"xsa 없음: {XSA} (vivado/build_bd.tcl 먼저)")

client = vitis.create_client()
client.set_workspace(path=WS)

if os.path.isdir(os.path.join(WS, PLATFORM)):
    platform = client.get_component(name=PLATFORM)
    print("플랫폼 하드웨어 갱신:", XSA)
    platform.update_hw(hw_design=XSA)
else:
    print("플랫폼 생성:", XSA)
    platform = client.create_platform_component(name=PLATFORM, hw_design=XSA, os="standalone",
                                                cpu="ps7_cortexa9_0", domain_name=DOMAIN)
platform.build()
xpfm = os.path.join(WS, PLATFORM, "export", PLATFORM, PLATFORM + ".xpfm")

if os.path.isdir(os.path.join(WS, APP)):
    app = client.get_component(name=APP)
    # 앱 실행 설정은 _ide/의 비트스트림·ps7_init 복사본을 쓰므로 새 플랫폼 것으로 바꿔 둠
    hw = os.path.join(WS, PLATFORM, "export", PLATFORM, "hw", "sdt")
    for src, dst in (("dpu_bd_wrapper.bit", "bitstream"), ("ps7_init.tcl", "psinit")):
        if os.path.exists(os.path.join(hw, src)):
            os.makedirs(os.path.join(WS, APP, "_ide", dst), exist_ok=True)
            shutil.copy(os.path.join(hw, src), os.path.join(WS, APP, "_ide", dst, src))
else:
    print("앱 생성:", APP)
    app = client.create_app_component(name=APP, platform=xpfm, domain=DOMAIN, template="empty_application")

def fix_launch():
    """Vitis가 HLS 드라이버의 dpu_top.tcl을 ps7 초기화 스크립트로 잡는 경우가 있어 ps7_init.tcl로 고정"""
    p = os.path.join(WS, APP, "_ide", "launch.json")
    if not os.path.exists(p):
        return
    s = open(p, encoding="utf8").read()
    s = re.sub(r'("ps7InitTclFile": "[^"]*psinit\\\\)[^"\\]+\.tcl"', r'\1ps7_init.tcl"', s)
    open(p, "w", encoding="utf8", newline="").write(s)


set_user_config()
fix_launch()
app.build()
print("elf:", os.path.join(WS, APP, "build", APP + ".elf"))
vitis.dispose()
