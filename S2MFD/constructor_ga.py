# TODO これを関数にして呼び出せるようにしたい
if __name__ == '__main__':
    # 最初の世代の人口決定
    defunction_initial_population: List[DefunctionProblem] = \
        [DefunctionProblem.make_random_instance() for _ in range(30)]
    ga: GeneticAlgorithm = GeneticAlgorithm(
        initial_population=defunction_initial_population,
        threshold=0.95,
        max_generations=1000,
        mutation_probability=0.2,
        crossover_probability=0.5,
        selection_type=GeneticAlgorithm.SELECTION_TYPE_TOURNAMENT)
    _ = ga.run_algorithm()

"""
コンストラクタに関してはアルゴリズムで必要な「最初の個体群」「アルゴリズムの停止判定に使われる評価関数のしきい値」
「世代数の上限」「変異確率」「交叉確率」「ルーレット選択などの選択方式」を指定しています。
"""