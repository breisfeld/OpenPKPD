; OpenPKPD NSIS installer script.
; Called by build_installer.py if makensis is on PATH.
; Defines expected on command line:
;   /DVERSION=x.y.z
;   /DDIST_DIR=<path to PyInstaller collected dir>
;   /DOUTPUT_DIR=<destination dir for installer .exe>

!define APP_NAME    "OpenPKPD"
!define PUBLISHER   "OpenPKPD Contributors"
!define APP_URL     "https://github.com/openpkpd/openpkpd"
!define REG_KEY     "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

Name            "${APP_NAME} ${VERSION}"
OutFile         "${OUTPUT_DIR}\OpenPKPD-${VERSION}-windows-x64-setup.exe"
InstallDir      "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "${REG_KEY}" "InstallLocation"

RequestExecutionLevel admin
SetCompressor    /SOLID lzma

;--- Pages ---
!include "MUI2.nsh"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${DIST_DIR}\..\..\..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

;--- Install ---
Section "MainSection" SEC01
  SetOutPath "$INSTDIR"
  File /r "${DIST_DIR}\*.*"

  ; Start Menu shortcut
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" \
                  "$INSTDIR\openpkpd-gui.exe"
  CreateShortcut  "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"   \
                  "$INSTDIR\uninstall.exe"

  ; Desktop shortcut
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\openpkpd-gui.exe"

  ; Add CLI to PATH via registry
  WriteRegExpandStr HKLM \
    "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" \
    "Path" "$INSTDIR;$%Path%"
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000

  ; Uninstall registry entry
  WriteRegStr   HKLM "${REG_KEY}" "DisplayName"      "${APP_NAME} ${VERSION}"
  WriteRegStr   HKLM "${REG_KEY}" "DisplayVersion"   "${VERSION}"
  WriteRegStr   HKLM "${REG_KEY}" "Publisher"        "${PUBLISHER}"
  WriteRegStr   HKLM "${REG_KEY}" "URLInfoAbout"     "${APP_URL}"
  WriteRegStr   HKLM "${REG_KEY}" "InstallLocation"  "$INSTDIR"
  WriteRegStr   HKLM "${REG_KEY}" "UninstallString"  "$INSTDIR\uninstall.exe"
  WriteRegDWORD HKLM "${REG_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${REG_KEY}" "NoRepair" 1

  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

;--- Uninstall ---
Section "Uninstall"
  Delete "$DESKTOP\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"
  RMDir  /r "$INSTDIR"
  DeleteRegKey HKLM "${REG_KEY}"
SectionEnd
