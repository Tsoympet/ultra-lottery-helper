; ultra_lottery_helper.iss — Native Desktop build

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
AppId={{F23A50E3-89F7-4D71-9F91-ULH-DESKTOP}}
AppName=Ultra Lottery Helper (Desktop)
AppVersion={#MyAppVersion}
AppPublisher=Tsoympet
AppPublisherURL=https://github.com/Tsoympet/ultra-lottery-helper
AppSupportURL=https://github.com/Tsoympet/ultra-lottery-helper/issues
DefaultDirName={autopf}\UltraLotteryHelper
DefaultGroupName=Ultra Lottery Helper (Desktop)
UninstallDisplayIcon={app}\ultra_lottery_helper.exe
SetupIconFile=assets\icon.ico
OutputDir=dist_installer
OutputBaseFilename=UltraLotteryHelperInstaller_{#MyAppVersion}
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2
SolidCompression=yes
LicenseFile=LICENSE.txt
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
Name: "{group}\Ultra Lottery Helper (Desktop)"; Filename: "{app}\ultra_lottery_helper.exe"; IconFilename: "{app}\assets\icon.ico"
; Desktop (προαιρετικό)
Name: "{commondesktop}\Ultra Lottery Helper (Desktop)"; Filename: "{app}\ultra_lottery_helper.exe"; IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon
; Quick Launch (παλιό αλλά διατηρείται)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Ultra Lottery Helper (Desktop)"; Filename: "{app}\ultra_lottery_helper.exe"; Tasks: quicklaunchicon
; Uninstall
Name: "{group}\Uninstall Ultra Lottery Helper (Desktop)"; Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "Create a Quick Launch shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\ultra_lottery_helper.exe"; Description: "Launch Ultra Lottery Helper (Desktop)"; Flags: nowait postinstall skipifsilent
