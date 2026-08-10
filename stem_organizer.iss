; STEM organizer — Inno Setup one-installer (Phase 3 ONNX build)
;
; Prerequisites:
;   1. Run build.bat  →  dist\STEM-organizer\
;   2. Install Inno Setup 6: https://jrsoftware.org/isinfo.php
;   3. Compile this script (ISCC stem_organizer.iss) or open in Inno GUI.
;
; Ships the onedir PyInstaller output. ONNX weights are already under
; dist\STEM-organizer\ (copied by build.bat). ffmpeg is bundled when present
; in dist, otherwise first-run helpers can fetch it.

#define MyAppName "STEM organizer"
#define MyAppVersion "1.0.8"
#define MyAppPublisher "STEM organizer"
#define MyAppExeName "STEM-organizer.exe"
#define MyAppSource "dist\STEM-organizer"

; --- Model weights: fetched from a GitHub Release at install time (not bundled) ---
; The 8 .onnx files are excluded from the installer and downloaded during
; install from the GitHub release tagged below.
; Sidecars (.json/.csv, ~30 KB total) are tiny and stay bundled.
;
; RELEASE WORKFLOW (once — weights are version-stable):
;   1. Tag a release on GitHub:  models  (= ModelsReleaseTag below; no app version).
;   2. Upload the 8 .onnx files as release assets.
;      Demucs: ship the dynamic-batch graph as asset ``htdemucs.batch.onnx``;
;      install DestName is still ``htdemucs.onnx`` (resolver + session cache).
;   3. Asset URL pattern:  https://github.com/<owner>/<repo>/releases/download/<tag>/<file>
;   4. Override the base URL at compile time if you fork/move:
;        iscc /dModelsBaseUrl="https://.../releases/download/models" stem_organizer.iss
;   The release may also host retired UMX-L/X-UMXL/SCNet/BS-RoFormer weights
;   (``_``-prefixed) as backup — they are NOT used here; keep them on the release.
;   Recompute Hash/ExternalSize only if a weight file actually changes.
#ifndef ModelsBaseUrl
  #define ModelsRepo    "bascurtiz/STEM-organizer-models"
  #define ModelsReleaseTag "models"
  #define ModelsBaseUrl    "https://github.com/" + ModelsRepo + "/releases/download/" + ModelsReleaseTag
#endif

; Compile-time only — never check this path at install time on the user's PC.
#if !DirExists(MyAppSource)
  #error Build output not found: dist\STEM-organizer — run build.bat first, then compile.
#endif

[Setup]
AppId={{8F3C2A1B-9E4D-4B7A-A6C1-STEMORG-PY6ONNX}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=dist\installer
OutputBaseFilename=STEM-organizer-setup-{#MyAppVersion}
SetupIconFile=logo.ico
WizardImageFile=wizard-image.bmp
WizardSmallImageFile=wizard-small.bmp
WizardImageStretch=yes
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Full installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
; The app itself is always installed. The "models" component downloads the
; ONNX weights from a GitHub release. A reinstalling user (or one
; with no internet) can uncheck it to skip the download — the app will prompt
; for missing models per-feature at runtime. Default: checked (download).
Name: "app"; Description: "{#MyAppName} (required)"; Types: full custom; Flags: fixed
Name: "models"; Description: "Download models (required for stem separation & tagging)"; Types: full; Flags: checkablealone

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Entire PyInstaller onedir (exe + _internal + taggers + demucs_onnx.py),
; MINUS the large .onnx weights — those are downloaded at install time from a
; GitHub release (declarative `external download` entries below) and only when
; the user keeps the "models" component checked. Tiny sidecars (.json/.csv,
; ~30 KB) always ship bundled with the onedir.
Source: "{#MyAppSource}\*"; Components: app; DestDir: "{app}"; Excludes: "\models\htdemucs.onnx,\models\htdemucs.batch.onnx,\models\cnn14.onnx,\models\stem_cnn6.onnx,\models\maest_discogs519.onnx,\models\discogs-effnet-bsdynamic-1.onnx,\models\gender-discogs-effnet-1.onnx,\models\vocal_reverb.onnx,\models\nf50-q05-221125.onnx"; Flags: ignoreversion recursesubdirs createallsubdirs

; --- Model weights: downloaded + SHA-256 verified at install time ---
; Only installed when the "models" component is selected (default: yes). A
; reinstalling user can uncheck it to skip the re-download. Inno shows a
; native progress bar and aborts that file on download failure or hash mismatch
; (tamper/corruption guard). ExternalSize is a byte hint; Hash is authoritative.
; RELEASE WORKFLOW: tag `models`, upload the weight files; recompute hashes only
; if a weight file changes.
; HTDemucs: StemSplit dynbatch graph as release asset htdemucs.batch.onnx → DestName htdemucs.onnx.
Source: "{#ModelsBaseUrl}/htdemucs.batch.onnx"; Components: models; DestName: "htdemucs.onnx"; DestDir: "{app}\models"; ExternalSize: 316446949; Hash: "8e9cfef49390c85093e6d557cf568748c35e04940c9b104564c4b723f5df072b"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/cnn14.onnx"; Components: models; DestName: "cnn14.onnx"; DestDir: "{app}\models"; ExternalSize: 327331890; Hash: "80310d45194dc603143e9b59631920254f6544917fce3d96170c4a5ea120bec1"; Flags: external download ignoreversion uninsneveruninstall
; Stem CNN6 instrument classifier ONNX (Rename Auto-detect + Classify, 11-class, ~24 MB).
Source: "{#ModelsBaseUrl}/stem_cnn6.onnx"; Components: models; DestName: "stem_cnn6.onnx"; DestDir: "{app}\models"; ExternalSize: 24158137; Hash: "df9e1e39f739d478aa6ba85dd3c7bd80d20e98c4f6e19fb7c69d3e0a437e3f0f"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/maest_discogs519.onnx"; Components: models; DestName: "maest_discogs519.onnx"; DestDir: "{app}\models"; ExternalSize: 348091011; Hash: "013e41a6b981be30c3a646c3581cac7b8dfcd35f2b3db01769b0681d3ffa0c8f"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/discogs-effnet-bsdynamic-1.onnx"; Components: models; DestName: "discogs-effnet-bsdynamic-1.onnx"; DestDir: "{app}\models"; ExternalSize: 18027718; Hash: "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/gender-discogs-effnet-1.onnx"; Components: models; DestName: "gender-discogs-effnet-1.onnx"; DestDir: "{app}\models"; ExternalSize: 514089; Hash: "e3e865d4bf36d4817f32ddab9452b2729f9e33a4d068d1c44ea44972a7999e91"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/vocal_reverb.onnx"; Components: models; DestName: "vocal_reverb.onnx"; DestDir: "{app}\models"; ExternalSize: 376812; Hash: "88fe45d4d16b3bbd8ff031095cc9a202f6ad0e71bf201147af6d4b07fd16cf08"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/nf50-q05-221125.onnx"; Components: models; DestName: "nf50-q05-221125.onnx"; DestDir: "{app}\models"; ExternalSize: 11483608; Hash: "9387f472000fc4cf7076668360fd35e918a3d32443db13144ac741fbfe604f6a"; Flags: external download ignoreversion uninsneveruninstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// --- Uninstall: ask whether to wipe leftover {app} data ---
// Large .onnx weights use `uninsneveruninstall`, so Inno leaves them (and any
// other leftovers: settings, caches, flac/mp3val tools, empty model dirs, CSVs)
// under {app} after the normal uninstall. At usPostUninstall we ask once:
//   Yes → DelTree the entire install folder (models + settings + tools + caches).
//   No  → leave leftovers for a faster reinstall (weights stay; settings too).
// Prompt whenever {app} still exists — not only when a sentinel weight is present.

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir: String;
  MsgRes: Integer;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  AppDir := ExpandConstant('{app}');
  // After Inno removes tracked files, {app} may still hold uninsneveruninstall
  // weights and/or leftover settings/caches/tools. Ask whenever the folder remains.
  if not DirExists(AppDir) then
    Exit;
  MsgRes := SuppressibleMsgBox(
    'Remove the entire STEM organizer install folder?' + #13#10#13#10 +
    'Yes deletes everything left under the install directory — downloaded models, ' +
    'settings, caches, tools (flac/mp3val), and other leftovers.' + #13#10#13#10 +
    'No keeps models and settings for a faster reinstall.',
    mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO);
  if MsgRes = IDYES then begin
    // usPostUninstall: Inno has finished removing tracked files; wipe the rest.
    if DelTree(AppDir, True, True, True) then
      Log('Uninstall: wiped install folder ' + AppDir)
    else
      Log('Uninstall: DelTree failed for ' + AppDir);
  end else begin
    Log('Uninstall: user chose to keep install-folder leftovers.');
  end;
end;

