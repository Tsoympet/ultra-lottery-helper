; ultra_lottery_helper.iss — Native Desktop build

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
WizardSmallImageFile=assets\banner_small.bmp
WizardImageFile=assets\banner.bmp
SignTool=mysigner
SignedUninstaller=yes
AppId={{F23A50E3-89F7-4D71-9F91-ULH-DESKTOP}}
AppName=Oracle Lottery Predictor
AppVersion=6.3.0
AppPublisher=Tsoympet
AppPublisherURL=https://github.com/Tsoympet/ultra-lottery-helper
AppSupportURL=https://github.com/Tsoympet/ultra-lottery-helper/issues
DefaultDirName={pf}\Oracle Lottery Predictor
DefaultGroupName=Oracle Lottery Predictor (Desktop)
UninstallDisplayIcon={app}\ultra_lottery_helper.exe
SetupIconFile=assets\icon.ico
OutputDir=dist_installer
OutputBaseFilename=OracleLotteryPredictor-Setup
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2
SolidCompression=yes
LicenseFile=EULA.rtf
PrivilegesRequired=admin
WizardStyle=modern

[Files]
; κύριο exe
Source: "dist\ultra_lottery_helper.exe"; DestDir: "{app}"; Flags: ignoreversion
; assets
Source: "assets\icon.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
; ιστορικά δεδομένα (αν υπάρχουν)
Source: "data\history\tzoker\*"; DestDir: "{app}\data\history\tzoker"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "data\history\lotto\*"; DestDir: "{app}\data\history\lotto"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "data\history\eurojackpot\*"; DestDir: "{app}\data\history\eurojackpot"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\data\history\tzoker"
Name: "{app}\data\history\lotto"
Name: "{app}\data\history\eurojackpot"
Name: "{app}\exports\tzoker"
Name: "{app}\exports\lotto"
Name: "{app}\exports\eurojackpot"
Name: "{app}\assets"

[Icons]
; Start Menu
Name: "{group}\Oracle Lottery Predictor (Desktop)"; Filename: "{app}\ultra_lottery_helper.exe"; IconFilename: "{app}\assets\icon.ico"
; Desktop (προαιρετικό)
Name: "{commondesktop}\Oracle Lottery Predictor (Desktop)"; Filename: "{app}\ultra_lottery_helper.exe"; IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon
; Quick Launch (παλιό αλλά διατηρείται)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Oracle Lottery Predictor (Desktop)"; Filename: "{app}\ultra_lottery_helper.exe"; Tasks: quicklaunchicon
; Uninstall
Name: "{group}\Uninstall Oracle Lottery Predictor (Desktop)"; Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "Create a Quick Launch shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\ultra_lottery_helper.exe"; Description: "Launch Oracle Lottery Predictor (Desktop)"; Flags: nowait postinstall skipifsilent

[SignTools]
Name: "mysigner"; Command: "signtool sign /fd SHA256 /a /tr http://timestamp.digicert.com /td sha256 $f"

[Files]
Source: "dist\\OracleLotteryPredictor\\*"; DestDir: "{app}"; Flags: recursesubdirs
