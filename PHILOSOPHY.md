# Philosophy

swift-lm の設計の根底にある確信を記述する。CLAUDE.md は **何をするか／してはならないか** のルールを述べる。本書は **なぜそう設計したか** を述べる。

---

## 1. swift-lm はコンパイラである

LLM 推論は伝統的に「フレームワーク」として設計されてきた。PyTorch、MLX、llama.cpp はいずれも、モデル定義を実行時に解釈し、テンソル演算を逐次ディスパッチする。これは**インタプリタ的アプローチ**である。

swift-lm はこの選択をしない。swift-lm は **HuggingFace bundle を入力とし、Metal kernel を出力するコンパイラ** である。

これを選ぶ理由:

- Apple Silicon の実測値で、decode の GPU 時間の **~85% が barrier 同期** に消える。kernel の計算時間そのものは支配的ではない
- 個別 kernel の高速化（matmul2d、AMX、ベクトル化）は限界がある — 全てが memory bandwidth bound だから
- **dispatch 数を削減する以外に、根本的な高速化レバーは存在しない**
- dispatch 数の削減は、隣接する fragment を融合する **静的解析** でしか実現できない

インタプリタは個別演算を最適化できる。コンパイラはグラフ全体を見て **演算の境界を消す** ことができる。これが swift-lm の選択である。

## 2. モデルはコードではなくデータである

swift-lm が消費者に要求するのは、**HuggingFace repo ID 1 つ** だけである。Swift で書かれたモデル定義クラスを書く必要はない。`config.json` + `safetensors` + `tokenizer.json` が正規入力。

理由:

- モデル定義をコードで持つと、新モデル対応のたびに Swift コードを書き、コンパイルし、リリースする必要がある
- データで持てば、新モデルの追加は **モデル宣言の追加** だけで済む。コアコードは変更不要
- HuggingFace が事実上のモデル流通標準である以上、その形式を受け入れることは流通コストを最小化する

これは「Swift API として表現力豊かなモデル DSL を提供する」という選択を **意図的に放棄している**。表現力の代わりに、**流通可能性とゼロコスト追加** を選んだ。

## 3. 関心の分離は 3 軸である

多くの推論フレームワークは「モデル構造」と「実行」の 2 軸で設計されている。swift-lm は **3 軸** に分ける。

| 軸 | モジュール | 関心事 |
|---|---|---|
| **WHAT** (構造) | LMIR | モデルの計算グラフ。Attention の存在、層数、次元 |
| **WHY** (意図) | InferencePolicy | デプロイメント判断。KV cache 量子化、最大シーケンス長 |
| **HOW** (実装) | MetalCompiler | kernel 選択、buffer 配置、dispatch plan |

`InferencePolicy` を独立した第三の軸に切り出したことが重要である。これは「IR でも compiler 内部でもない」**消費者の意図** を表す。同じ IR と同じ compiler でも、policy が違えば異なる Metal kernel が出る。

この分離を曖昧にすると、IR にデプロイメント設定が漏れ込み、compiler に固有名がハードコードされ、消費者が compiler 内部を触らないと挙動を変えられなくなる。

## 4. Fragment は自己記述的である。Compiler は無知である

Compiler は fragment の具象型を知らない。`FusionContract`、`kernelBody()`、`decodeBindings()` という **protocol interface のみ** で全ての判断を行う。

新しい fragment を追加するとき、compiler のコードは 1 行も変わらない。

これは Open-Closed Principle の極致である:

- compiler が `if fragment is XxxFragment` と書いたら負け
- `DispatchKind` に新 case を追加して fusion パターンを増やしたら負け
- `MetalSourceGenerator` に `generateFusedXxx()` を追加したら負け

正しい拡張は: fragment が自分の `FusionContract` を宣言し、compiler が機械的に判定すること。

なぜここまで厳格にするか:

- swift-lm は将来 **数十種の fragment と数百のモデル** を扱うことになる
- compiler 側に分岐を追加する設計は、組み合わせ爆発と保守コスト爆発で破綻する
- 唯一の防御策は、**新しいケースが compiler に触れさせない** という不変条件

## 5. HuggingFace が唯一の正である

各モデルの計算ロジックは HuggingFace の `modeling_*.py` の `forward()` を正とする。**swift-lm の既存実装・既存テストを正しいと仮定してはならない。**

これは強い主張である:

- 「他のモデルが動いているから、このモデルも同じ実装で動くだろう」は禁止
- Gemma2 の RMSNorm 式を Gemma4 に流用しない。Qwen3.5 のパラメータを Gemma4 に流用しない
- swift-lm のコード同士の比較で「正しいはず」と推測しない

正しさの判定は **常に外部参照 (HuggingFace Python の中間値)** との比較で行う。内部一貫性は正しさの証明にならない。

これを徹底する理由は実体験にある: 内部比較で「正常」と判定した実装が、HuggingFace と比較すると実は全層が壊れていた、というケースが現実に起きる。**全層が壊れている場合、内部比較は壊れたものと壊れたものを比較して合格判定を出してしまう。**

## 6. Probe First — 観測が静的解析に先立つ

出力が壊れたとき、最初にやることはコードを読むことではない。**Probe で壊れた場所を観測する** ことである。

順序:

1. Layer Probe で全層の hidden state を検査 — 障害層を特定する
2. 障害層の step を精査 — 壊れた kernel を特定する
3. HuggingFace Python の中間値と比較する
4. **その後で初めて** kernel source を読む

なぜこの順序か:

- 静的解析から始めると、コードの「正しそうな」部分を見て安心してしまう
- 実行時の値を見ずに仮説を立てると、間違った場所を直してしまう
- 「機能を無効化して切り分ける」ような診断は、正常パスを壊すので診断にならない

これは経験則である。出力が壊れた状態でコードを読んでも、たいていは何も発見できない。値を観測して初めて、「ここが壊れている」という事実から逆算できる。

## 7. Silent Fallback は禁止する

エラーや異常時に、デフォルト値や代替パスへ黙って切り替える処理を一切認めない。

具体例:

- `MetalCompilable` に対応しない `OperationAttributes` は `fatalError` する。代替実装を探さない
- `KVCacheSpecification.maximumSequenceLength` にデフォルト値を持たせない
- `try?` でエラーを握りつぶさない
- config.json に必須項目が欠けていたら補完せずエラー

理由:

- silent fallback は「動いているように見えるが間違った結果を出す」状態を作る
- 一度 fallback パスが有効になると、それが本来の挙動だと誤認される
- 失敗は **明示的に報告** し、呼び出し元に判断を委ねる

これは「親切な API」より「予測可能な API」を選ぶという思想である。

## 8. 性能を追求する

swift-lm の存在理由は性能である。**「Apple Silicon 上での最速 LLM 推論」** が project goal であり、それ以外の価値（教育性、ポータビリティ、表現力豊かなモデル DSL、親切なエラーハンドリング）はすべてこれに従属する。

性能のためなら以下を厭わない:

- 既存実装を捨てて compiler パイプラインから書き直すこと
- API の見た目を諦めて低レベル制御を露出させること
- 特定のハードウェア（Apple Silicon）に深くロックインすること
- 数十モデル分の検証コストを払うこと
- フレームワークではなくコンパイラを実装するという複雑さを引き受けること

ただし、**性能を追求するからこそ、性能を正しく測れる状態を維持する** ことは必須となる。「Output Verification Gate」と呼ぶ手続きが存在するのはこのためである。

- 壊れた出力の生成速度を性能とは呼ばない — ノイズを速く出しても価値はゼロ
- benchmark の数字は、出力が正しいことが確認された状態でのみ意味を持つ
- thinking E2E で `answer` が空のまま通った run の throughput は無意味
- 1 つのモデル / 1 つの経路で通った結果を、別モデル / 別経路の性能保証に使わない

これは「正しさが性能に勝る」という主張ではない。**性能こそが目的だからこそ、性能の測定が嘘であってはならない** という主張である。壊れた状態で出したベンチマーク数値は自分を欺く嘘であり、性能を追求する者が最も避けるべき状態である。

## 9. Apple Silicon は前提であり、target ではない

swift-lm は portable framework ではない。Apple Silicon の unified memory、TBDR、Metal 4 を **前提として** 設計されている。

具体的には:

- Metal 3 ではなく Metal 4 を優先する
- Apple Silicon の lossless compression が効くように `private` buffer を多用する
- macOS / iOS / visionOS 26.1+ を最低バージョンとする

これは「より広いプラットフォームをサポートする」という選択を放棄している。代わりに、**特定のアーキテクチャに対する最適化の深さ** を選んだ。

「いつか CUDA / TPU でも動かしたい」という願望のために IR を抽象化することはあっても、**実装に妥協を持ち込まない**。LMIR が backend 非依存なのは、将来別 backend を作りたいからではなく、**Metal 実装の責務境界を明確にするため** である。

## 10. 我々が拒否するもの

明示的に拒否する設計選択:

- ❌ 個別 kernel の手書き最適化 (matmul2d、AMX 直書き) — 効果が薄く保守コストが高い
- ❌ 手書きの fused kernel (`generateFusedCopyRMSNorm` 等) — automatic fusion で代替する
- ❌ compiler 内の `switch fragment.kind` 分岐 — Open-Closed 違反
- ❌ モデル別 Swift コード生成 — データで表現可能なことをコードで表現しない
- ❌ "それっぽく動くテスト" — 出力検証が伴わないテストはリリース根拠にならない
- ❌ 内部一貫性に基づく正しさ判定 — 外部参照 (HuggingFace) との比較を必ず行う
- ❌ デフォルト値による silent fallback — 失敗は明示的に
- ❌ ベンチマーク優先のリファクタリング — correctness gate を通っていない最適化は議論しない

---

## 結語

swift-lm の設計は、**いくつかの強い拒否** の上に成り立っている。何でもサポートする、何でも親切にハンドリングする、何でも portable にする、という方向は明示的に選んでいない。

その代わりに、**Apple Silicon 上で、HuggingFace 標準形式から、最速で正しい LLM 推論を出す** という 1 点に集中している。コンパイラとして設計されているのは、その 1 点を達成する手段が他にないからである。
