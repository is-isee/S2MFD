"""描画用関数"""
def make_graph(x,y,label,label_x,label_y,file_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))  # グラフのサイズを調整
    plt.plot(x,y,'r',label=label)
    plt.xlabel(label_x,fontsize=20)
    plt.ylabel(label_y,fontsize=20)
    # 軸のメモリを細かく設定
    plt.xticks(fontsize=14)  # x軸の数値サイズを調整
    plt.yticks(fontsize=14)  # y軸の数値サイズを調整
    # グリッドを追加して見やすく
    plt.grid(True, linestyle='--', alpha=0.7)
    # 凡例を表示
    plt.legend(fontsize=14, loc='upper right')  # 凡例を右上に固定
    # グラフを保存
    plt.savefig(file_name, dpi=300)
    plt.clf()