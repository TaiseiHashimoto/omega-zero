# omega-zero
AlphaZeroのオセロバージョン

## mediator.py について
2つのプログラムを自動的に対戦させるプログラムです.  
(当然ながら)各プログラムは一定の入出力フォーマットに従っていただく必要があります.  
以下のフォーマットに従ったにもかかわらず動作しない場合はお知らせください.  

1. はじめに色(黒/白)を決める際のプロンプトは "@ [b]lack / [w]hite ?" とする  
1. 色の入力は "b" / "w" で受け取る  
1. 次の手のプロンプトは "@ action ?" とする  
1. 選択した手を出力する際は "@ action \w+" とする (\w+が "a1" などの手に対応する)  
1. パスは "pass" と記述し, 他の手と同様に扱う
1. 最後に結果を "@ result : black=[0-9]+ white=[0-9]+" と出力する  
1. meditatorに読ませるプロンプト・出力はすべて最後に改行する  

空白にも十分注意してください.  
また, プログラムが違法な打つ場合などは考えていません.  
