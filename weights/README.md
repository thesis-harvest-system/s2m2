# S2M2 事前学習重み (Google Drive 管理)

`stereo_node` が読む事前学習重み **`pretrain_weights/CH384NTR3.pth` (約 1.6GB)** は
git に含めない。理由:

- このリポジトリ `thesis-harvest-system/s2m2` は本家 S2M2 の **public fork** で、GitHub は
  fork への **git-lfs オブジェクトのアップロードを許可しない** (`can not upload new objects
  to public fork`)。
- 1.6GB は GitHub の通常制限 (100MB/file) も無料 LFS 枠 (1GB) も超える。

そのため重みは **Google Drive** で配布し、`.gitignore` の `*.pth` で git 追跡から外している。
中身は **S2M2 のデフォルト事前学習重みそのまま**(fine-tune 等はしていない)で、公開しても
問題ないため **リンク公開**(下記 `gdown` で誰でも取得可)で配っている。

配置先 (この相対パス固定。`stereo` パッケージの share へインストールされる):

```
weights/pretrain_weights/CH384NTR3.pth
```

---

## ダウンロード手順

`gdown` を使う (`pip install gdown`)。`<FILE_ID>` は下記「共有リンクの ID」を参照。

```bash
cd <このリポジトリのルート>          # .../s2m2
mkdir -p weights/pretrain_weights
gdown "https://drive.google.com/uc?id=1tVSD9xdiqSabe8XvozSftFMGZle3Tfl9" \
      -O weights/pretrain_weights/CH384NTR3.pth
```

ダウンロード後にサイズを確認 (約 1.6GB):

```bash
ls -lh weights/pretrain_weights/CH384NTR3.pth
```

> `gdown` が「ウイルススキャンできない大容量ファイル」警告で失敗する場合は
> `gdown --fuzzy "<共有リンクそのまま>" -O weights/pretrain_weights/CH384NTR3.pth` を試す。

### 共有リンクの ID

Google Drive の共有リンク `https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing` の
`<FILE_ID>` 部分を使う。

- **FILE_ID**: `1tVSD9xdiqSabe8XvozSftFMGZle3Tfl9`
- 共有設定: 「リンクを知っている全員 (閲覧者)」(`rclone link` で設定済み)。

---

## アップロード手順 (重みを更新・差し替えるとき)

### 方法 A: ブラウザ (手軽)

1. [Google Drive](https://drive.google.com/) を開き、配布用フォルダ
   (例: `ynl/s2m2_weights`) を作る。
2. `CH384NTR3.pth` をドラッグ&ドロップでアップロード。
3. 右クリック → 「共有」→ 一般的なアクセスを「リンクを知っている全員」=「閲覧者」に。
4. 「リンクをコピー」して得た URL の `<FILE_ID>` を上の **FILE_ID** 欄へ記入し、
   この変更をコミットする。

### 方法 B: rclone (CLI / 自動化向き)

```bash
# 初回のみ: Google Drive を remote として設定 (対話)
rclone config        # n → 名前(例 gdrive) → "drive" を選択 → 認証

# アップロード
rclone copy weights/pretrain_weights/CH384NTR3.pth gdrive:s2m2_weights/

# 共有リンク(FILE_ID 取得用)
rclone link gdrive:s2m2_weights/CH384NTR3.pth
```

得られたリンクの `<FILE_ID>` を上の **FILE_ID** 欄へ記入してコミットする。
