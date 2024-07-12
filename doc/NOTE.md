# 2024/07/12

## 周波数積分関連の不具合解決に関するメモ

- やりたいこと：バンドパスフィルターによる周波数積分の実装
- 問題点：周波数積分の結果がViewWaveと比較して異なる
    - ViewWaveは`次数`と書かれているので、`Butterworth`フィルターを使っていると思われる
    - ソフトウェア起動時のデフォルト値は、低域遮断周波数が`0.1Hz`、次数が`4`である
- `scipy.signal.butter`関数だと、同じ値は得られない?
    - ちなみに`btype`は`high`でも`highpass`でも構わない
        - [ソースコード](https://github.com/scipy/scipy/blob/87c46641a8b3b5b47b81de44c07b840468f7ebe7/scipy/signal/_filter_design.py#L5599)
    - いやこの関数の`N`と`Wn`にはそれぞれオーダーと、Critical Frequencyを入力する
        - > For a Butterworth filter, this is the point at which the gain drops to 1/sqrt(2) that of the passband (the “-3 dB point”).
        - つまり、`Wn`は`-3dB`の周波数を指定する
    - `analog=True`にすると、アナログフィルターの設計になる
        - 2つの返り値`b`と`a`は、それぞれ分子と分母の係数を表す
        - 具体的な計算式は以下の通り
            - $H(s) = \frac{b[0]s^{N} + b[1]s^{N-1} + \cdots + b[N]}{a[0]s^{N} + a[1]s^{N-1} + \cdots + a[N]}$
    - Viewwaveではソースコードを読むと、周波数領域で作用させているっぽい
    - `output="ba"`は後方互換性のために残されている
        - これを使うと、`b`と`a`が返り値として得られる
    - ドキュメントとしては、`sos`を使うことが推奨されている
