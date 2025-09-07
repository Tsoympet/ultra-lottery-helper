[Setup]
AppName=Ultra Lottery Helper
AppVersion={#MyAppVersion}
AppPublisher=Your Name
AppPublisherURL=https://github.com/yourname/ultra-lottery-helper
AppSupportURL=https://github.com/yourname/ultra-lottery-helper/issues
DefaultDirName={pf}\UltraLotteryHelper
DefaultGroupName=Ultra Lottery Helper
UninstallDisplayIcon={app}\ultra_lottery_helper.exe
SetupIconFile=assets\icon.ico
OutputDir=dist_installer
OutputBaseFilename=UltraLotteryHelperInstaller_{#MyAppVersion}
ArchitecturesInstallIn64BitMode=x64
Compression=lzma2
SolidCompression=yes
LicenseFile=LICENSE.txt
PrivilegesRequired=admin

[Files]
Source: "dist\ultra_lottery_helper.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\icon.ico"; DestDir: "{app}\assets"; Flags: ignoreversion
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
Name: "{group}\Ultra Lottery Helper"; Filename: "{app}\ultra_lottery_helper.exe"; IconFilename: "{app}\assets\icon.ico"
Name: "{commondesktop}\Ultra Lottery Helper"; Filename: "{app}\ultra_lottery_helper.exe"; IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\Ultra Lottery Helper"; Filename: "{app}\ultra_lottery_helper.exe"; Tasks: quicklaunchicon
Name: "{group}\Uninstall Ultra Lottery Helper"; Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "Create a Quick Launch shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked
