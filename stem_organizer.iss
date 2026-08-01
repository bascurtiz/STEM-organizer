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
; The 8 .onnx files (~1.36 GB raw / ~435 MB compressed) are excluded from the
; installer and downloaded during install from the GitHub release tagged below.
; Sidecars (.json/.csv, ~30 KB total) are tiny and stay bundled.
;
; RELEASE WORKFLOW (once — weights are version-stable):
;   1. Tag a release on GitHub:  models  (= ModelsReleaseTag below; no app version).
;   2. Upload the 8 .onnx files as release assets (each < 2 GB).
;   3. Asset URL pattern:  https://github.com/<owner>/<repo>/releases/download/<tag>/<file>
;   4. Override the base URL at compile time if you fork/move:
;        iscc /dModelsBaseUrl="https://.../releases/download/models" stem_organizer.iss
;   Recompute Hash/ExternalSize only if a weight file actually changes
;   (python _onnx_spike/hash_weights.py --patch).
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
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Full installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
; The app itself is always installed. The "models" component is the ~1.4 GB of
; ONNX weights, downloaded from a GitHub release. A reinstalling user (or one
; with no internet) can uncheck it to skip the download — the app will prompt
; for missing models per-feature at runtime. Default: checked (download).
Name: "app"; Description: "{#MyAppName} (required)"; Types: full custom; Flags: fixed
Name: "models"; Description: "Download AI models (~1.4 GB, required for stem separation & tagging)"; Types: full; Flags: checkablealone

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Entire PyInstaller onedir (exe + _internal + taggers + demucs_onnx.py),
; MINUS the 8 large .onnx weights — those are downloaded at install time from a
; GitHub release (declarative `external download` entries below) and only when
; the user keeps the "models" component checked. Tiny sidecars (.json/.csv,
; ~30 KB) always ship bundled with the onedir.
Source: "{#MyAppSource}\*"; Components: app; DestDir: "{app}"; Excludes: "\models\htdemucs.onnx,\panns_tagger\models\cnn14.onnx,\instrument_tagger\models\passt_openmic.onnx,\genre_gender_tagger\models\maest_discogs519.onnx,\genre_gender_tagger\models\discogs-effnet-bsdynamic-1.onnx,\genre_gender_tagger\models\gender-discogs-effnet-1.onnx,\genre_gender_tagger\models\vocal_reverb.onnx,\key_tagger\checkpoints\nf50-q05-221125.onnx"; Flags: ignoreversion recursesubdirs createallsubdirs

; --- Model weights: downloaded + SHA-256 verified at install time ---
; Only installed when the "models" component is selected (default: yes). A
; reinstalling user can uncheck it to skip the ~1.4 GB re-download. Inno shows a
; native progress bar and aborts that file on download failure or hash mismatch
; (tamper/corruption guard). ExternalSize is a byte hint; Hash is authoritative.
; RELEASE WORKFLOW: tag `models`, upload the 8 files; recompute hashes only
; if a weight file changes (see _onnx_spike/hash_weights.py).
Source: "{#ModelsBaseUrl}/htdemucs.onnx"; Components: models; DestName: "htdemucs.onnx"; DestDir: "{app}\models"; ExternalSize: 316446953; Hash: "68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/cnn14.onnx"; Components: models; DestName: "cnn14.onnx"; DestDir: "{app}\panns_tagger\models"; ExternalSize: 327331890; Hash: "80310d45194dc603143e9b59631920254f6544917fce3d96170c4a5ea120bec1"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/passt_openmic.onnx"; Components: models; DestName: "passt_openmic.onnx"; DestDir: "{app}\instrument_tagger\models"; ExternalSize: 341564125; Hash: "16e85ea2fac40b9f3c211b96ccb393111ea281f8a9e0f1a40322f2f726303b7a"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/maest_discogs519.onnx"; Components: models; DestName: "maest_discogs519.onnx"; DestDir: "{app}\genre_gender_tagger\models"; ExternalSize: 348091011; Hash: "013e41a6b981be30c3a646c3581cac7b8dfcd35f2b3db01769b0681d3ffa0c8f"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/discogs-effnet-bsdynamic-1.onnx"; Components: models; DestName: "discogs-effnet-bsdynamic-1.onnx"; DestDir: "{app}\genre_gender_tagger\models"; ExternalSize: 18027718; Hash: "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/gender-discogs-effnet-1.onnx"; Components: models; DestName: "gender-discogs-effnet-1.onnx"; DestDir: "{app}\genre_gender_tagger\models"; ExternalSize: 514089; Hash: "e3e865d4bf36d4817f32ddab9452b2729f9e33a4d068d1c44ea44972a7999e91"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/vocal_reverb.onnx"; Components: models; DestName: "vocal_reverb.onnx"; DestDir: "{app}\genre_gender_tagger\models"; ExternalSize: 376812; Hash: "88fe45d4d16b3bbd8ff031095cc9a202f6ad0e71bf201147af6d4b07fd16cf08"; Flags: external download ignoreversion uninsneveruninstall
Source: "{#ModelsBaseUrl}/nf50-q05-221125.onnx"; Components: models; DestName: "nf50-q05-221125.onnx"; DestDir: "{app}\key_tagger\checkpoints"; ExternalSize: 11483608; Hash: "9387f472000fc4cf7076668360fd35e918a3d32443db13144ac741fbfe604f6a"; Flags: external download ignoreversion uninsneveruninstall

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// --- Uninstall: ask whether to keep or remove the downloaded AI models ---
// The 8 .onnx weights (~1.4 GB) were downloaded at install time. By default
// Inno's uninstall log would remove them silently. To let a reinstalling user
// KEEP them (so they don't re-download), we suppress auto-removal via the
// `nouninstallfiles`-equivalent (these are external-download files, tracked in
// the uninstall log) and ask once during uninstall.
//
// NOTE on `external download` tracking: Inno records successfully-installed
// external files in the uninstall log, so they ARE removed by default. To make
// the keep-choice meaningful we delete them ONLY if the user says yes here, and
// mark the weight [Files] entries so they're not auto-uninstalled. Because we
// can't conditionally un-track them at install time, we instead re-add them to
// the deletion set here only when the user confirms.

const
  // Dist-relative paths of the 8 downloaded weights (must match [Files] above).
  UNINSTALL_WEIGHTS =
    '\models\htdemucs.onnx|\panns_tagger\models\cnn14.onnx|\instrument_tagger\models\passt_openmic.onnx|\genre_gender_tagger\models\maest_discogs519.onnx|\genre_gender_tagger\models\discogs-effnet-bsdynamic-1.onnx|\genre_gender_tagger\models\gender-discogs-effnet-1.onnx|\genre_gender_tagger\models\vocal_reverb.onnx|\key_tagger\checkpoints\nf50-q05-221125.onnx';

function SplitAndDeleteWeights(const AppDir, PipeList: String): Integer;
// Splits PipeList on '|', deletes each existing file under AppDir. Returns count.
var
  S, P: String;
  Count: Integer;
begin
  Count := 0;
  S := PipeList;
  while (S <> '') do begin
    if Pos('|', S) > 0 then begin
      P := Copy(S, 1, Pos('|', S) - 1);
      Delete(S, 1, Pos('|', S));
    end else begin
      P := S;
      S := '';
    end;
    if (P <> '') and FileExists(AppDir + P) then begin
      if DeleteFile(AppDir + P) then
        Count := Count + 1;
    end;
  end;
  Result := Count;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir: String;
  MsgRes: Integer;
  Removed: Integer;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  AppDir := ExpandConstant('{app}');
  // Only ask if at least one weight is actually present.
  // (Cheap presence check via the Demucs weight — the largest, always expected.)
  if not FileExists(AppDir + '\models\htdemucs.onnx') then
    Exit;
  MsgRes := SuppressibleMsgBox(
    'Remove the downloaded AI models (~1.4 GB)?' + #13#10#13#10 +
    'Choose Yes to free the disk space.' + #13#10 +
    'Choose No to keep them for a faster reinstall.',
    mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO);
  if MsgRes = IDYES then begin
    Removed := SplitAndDeleteWeights(AppDir, UNINSTALL_WEIGHTS);
    Log('Uninstall: removed ' + IntToStr(Removed) + ' model weight file(s).');
  end else begin
    Log('Uninstall: user chose to keep model weights.');
  end;
end;

