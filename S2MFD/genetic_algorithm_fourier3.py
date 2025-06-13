from __future__ import annotations
from typing import TypeVar, List, Dict
from random import choices, random, randrange, shuffle
from heapq import nlargest
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
import random
import S2MFD
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
import time

# TODO: 変更箇所①
PARAMETER_NAMES = ['a0_s', 'a1_s', 'a2_s', 'a3_s', 'b1_s', 'b2_s', 'b3_s', 'omega_s',
                   'a0_u', 'a1_u', 'a2_u', 'a3_u', 'b1_u', 'b2_u', 'b3_u', 'omega_u']
def GA_defunction(cfg=None, parameter_file=None, datadir=None, startpoint=0, endpoint=0):
    """
    観測データのインプット
    """
    start_time = time.time()
    if datadir is None:
        print('You need to specify the datadir')
        return
    if cfg is None:
        if parameter_file is None:
            cfg = S2MFD.Cfg()
        else:
            cfg = S2MFD.Cfg(parameter_file)
    sim = S2MFD.Simulation(cfg)
    n1, Bpht, Apht, uu0t, so0t, nt, ndt, timet = sim.load_for_bisection(datadir)
    Sunspot_N, Sunspot_N2 = sim.pre_snumbers_energy(Bpht)
    
    """
    初期世代の生成、観測データをGAにインプット
    """
    defunction_initial_population: List[DefunctionProblem] = [
    DefunctionProblem.make_random_instance(
        parameter_file, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N
    ) for _ in range(30)  # 個体数
    ]
    
    """
    GAの設定と実行
    """
    ga: GeneticAlgorithm = GeneticAlgorithm(
        initial_population=defunction_initial_population,
        threshold=0.485,
        max_generations=1000,
        mutation_probability=0.3,
        crossover_probability=0.8,
        selection_type=GeneticAlgorithm.SELECTION_TYPE_TOURNAMENT,  # 選択方式
        crossover_type=GeneticAlgorithm.CROSSOVER_TYPE_UNIFORM,  # 交叉方式
        mutation_type=GeneticAlgorithm.MUTATION_TYPE_UNIFORM  # 突然変異方式
    )
    _ = ga.run_algorithm()
    end_time = time.time()
    print(f"GA実行時間: {end_time - start_time:.2f}秒")
class Chromosome(ABC):
    """
    染色体（遺伝的アルゴリズムの要素1つ分）を扱う抽象クラス。
    """

    @abstractmethod
    def get_fitness(self) -> float:
        """
        対象の問題に対する染色体の優秀さを取得する評価関数Y用の
        抽象メソッド。

        Returns
        -------
        fitness : float
            対象の問題に対する染色体の優秀さの値。高いほど問題に
            適した染色体となる。
            遺伝的アルゴリズムの終了判定などにも使用される。
        """
        ...

    @classmethod
    @abstractmethod
    def make_random_instance(cls) -> Chromosome:
        """
        ランダムな特徴（属性値）を持ったインスタンスを生成する
        抽象メソッド。

        Returns
        -------
        instance : Chromosome
            生成されたインスタンス。
        """
        ...


    def __lt__(self, other: Chromosome) -> bool:
        """
        個体間の比較で利用する、評価関数の値の小なり比較用の関数。

        Parameters
        ----------
        other : Chromosome
            比較対象の他の個体。

        Returns
        -------
        result_bool : bool
            小なり条件を満たすかどうかの真偽値。
        """
        return self.get_fitness() < other.get_fitness()
C = TypeVar('C', bound=Chromosome)
class GeneticAlgorithm:
    # 選択タイプの指定
    SelectionType = int
    SELECTION_TYPE_ROULETTE_WHEEL: SelectionType = 1
    SELECTION_TYPE_TOURNAMENT: SelectionType = 2
    SELECTION_TYPE_VARIABLE_TOURNAMENT: SelectionType = 3
    
    # 交叉タイプの指定
    CrossoverType = int
    CROSSOVER_TYPE_SINGLE_POINT: CrossoverType = 1
    CROSSOVER_TYPE_UNIFORM: CrossoverType = 2

    # 突然変異タイプの指定
    MutationType = int
    MUTATION_TYPE_UNIFORM: MutationType = 1
    MUTATION_TYPE_GAUSSIAN: MutationType = 2

    # コンストラクタの書き方
    def __init__(
            self, initial_population: List[C],
            threshold: float,
            max_generations: int, mutation_probability: float,
            crossover_probability: float,
            selection_type: SelectionType,
            crossover_type: CrossoverType,
            mutation_type: MutationType) -> None:
        """
        遺伝的アルゴリズムを扱うクラス。

        Parameters
        ----------
        initial_population : list of Chromosome
            最初の世代の個体群（染色体群）。
        threshold : float
            問題解決の判定で利用するしきい値。この値を超える個体が
            発生した時点で計算が終了する。
        max_generations : int
            アルゴリズムで実行する最大世代数。
        mutation_probability : float
            変異確率（0.0～1.0）。
        crossover_probability : float
            交叉確率（0.0～1.0）。
        selection_type : int
            選択方式。以下のいずれかの定数値を指定する。
            - SELECTION_TYPE_ROULETTE_WHEEL
            - SELECTION_TYPE_TOURNAMENT
        crossover_type : int
            交叉方式。以下のいずれかの定数値を指定する。
            - CROSSOVER_TYPE_SINGLE_POINT
            - CROSSOVER_TYPE_UNIFORM
        mutation_type : int
            突然変異方式。以下のいずれかの定数値を指定する。
            - MUTATION_TYPE_UNIFORM
            - MUTATION_TYPE_GAUSSIAN
        """
        self._population: List[Chromosome] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_probability: float = mutation_probability
        self._crossover_probability: float = crossover_probability
        self._selection_type: int = selection_type
        self._crossover_type: int = crossover_type
        self._mutation_type: int = mutation_type
    # =================================================================== #
    """ Def_Selection Methods """
    """ Roulette_Wheel_Selection """
    def _exec_roulette_wheel_selection(self) -> List[Chromosome]:
        """
        ルーレット選択を行い、交叉などで利用する2つの個体（染色体）を
        取得する。

        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。選択処理は評価関数
            （fitnessメソッド）による重みが設定された状態でランダムに抽出される。

        Notes
        -----
        評価関数の結果の値が負になる問題には利用できない。
        """
        weights: List[float] = [
            chromosome.get_fitness() for chromosome in self._population]
        selected_chromosomes: List[Chromosome] = choices(
            self._population, weights=weights, k=2)
        return selected_chromosomes
    
    """ Tournament_Selection """
    def _exec_tournament_selection(self) -> List[Chromosome]:
        """
        トーナメント選択を行い、交叉などで利用するための2つの個体
        （染色体）を取得する。

        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。トーナメント
            用に引数で指定された件数分抽出された中から上位の2つの個体が
            設定される。
        """
        participants_num: int = len(self._population) // 2
        participants: List[Chromosome] = choices(self._population, k=participants_num)
        # ====
        # ★ fitnessが未計算の場合に例外を防ぐ
        for participant in participants:
            if not hasattr(participant, '_fitness'):
                raise RuntimeError("参加者のfitnessが未計算です。")
        # ====
        
        selected_chromosomes: List[Chromosome] = nlargest(n=2, iterable=participants)
        return selected_chromosomes
    """ Variable Tournament_Selection """
    def _exec_variable_tournament_selection(self) -> List[Chromosome]:
        """
        可変的トーナメント選択を行い、交叉などで利用するための2つの個体
        （染色体）を入手する。
        
        Parameters
        ----------
        generation_idx : int
            現在の世代数。選択する個体の件数を世代数に応じて変化させる。
            
        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。トーナメント
            用に引数で指定された件数分抽出された中から上位の2つの個体が
            設定される。
        
        """
        generation_idx = len(self._population)  # 世代数を取得（仮定）
        base_size = len(self._population) // 4  # 初期サイズ（集団の1/4）
        max_size = len(self._population) // 2  # 最大サイズ（集団の1/2）
        tournament_size = min(base_size + generation_idx, max_size)
    
        # トーナメント参加者をランダムに選択
        participants: List[Chromosome] = choices(self._population, k=tournament_size)

        # fitnessが未計算の場合に例外を防ぐ
        for participant in participants:
            if not hasattr(participant, '_fitness'):
                raise RuntimeError("参加者のfitnessが未計算です。")

        # トーナメント内で最良の2個体を選択
        selected_chromosomes: List[Chromosome] = nlargest(n=2, iterable=participants)
        return selected_chromosomes
    # =================================================================== #
    """ def execute_crossover methods """
    def _exec_single_point_crossover(self, parents: List[Chromosome]) -> List[Chromosome]:
        """
        一点交叉を実行する。
        """
        from copy import deepcopy
        child_1 = deepcopy(parents[0])
        child_2 = deepcopy(parents[1])
        # 一点交叉の分割点をランダムに選択
        crossover_point = random.randint(1, len(PARAMETER_NAMES) - 1)

        # 分割点で属性を交互に交換
        for i in range(crossover_point, len(PARAMETER_NAMES)):
            attr = PARAMETER_NAMES[i]
            child_1.parameters[attr], child_2.parameters[attr] = child_2.parameters[attr], child_1.parameters[attr]
            
        print(f"single_point_crossover at {crossover_point}:")
        print(f"child_1 {child_1.parameters}")
        print(f"child_2 {child_2.parameters}")
        
        return [child_1, child_2]

    def _exec_uniform_crossover(self, parents: List[Chromosome]) -> List[Chromosome]:
        """
        一様交叉を実行する。
        """
        from copy import deepcopy
        child_1 = deepcopy(parents[0])
        child_2 = deepcopy(parents[1])
        # 一様交叉の例: ランダムに属性を交換
        for attr in PARAMETER_NAMES:
            if random.random() > 0.5:
                child_1.parameters[attr], child_2.parameters[attr] = child_2.parameters[attr], child_1.parameters[attr]
        print(f"uniform_crossover: child_1 {child_1.parameters}, child_2 {child_2.parameters}")
        return [child_1, child_2]
    # =================================================================== #
    # =================================================================== #
    """ def mutation methods """
    def _exec_uniform_mutation(self, chromosome: Chromosome) -> None:
        """
        一様突然変異を実行する。
        """
        target: str = random.choice(PARAMETER_NAMES)
        # before = getattr(chromosome, target)  # 変異前の値を取得
        before = chromosome.parameters[target]

        new_value = before * random.uniform(0.9, 1.1)
        # setattr(chromosome, target, new_value)
        chromosome.parameters[target] = new_value 

        # after = getattr(chromosome, target)  # 変異後の値を取得
        after = chromosome.parameters[target]
        print(f"uniform_mutation: {target} {before} -> {after}")

    def _exec_gaussian_mutation(self, chromosome: Chromosome) -> None:
        """
        ガウス分布に基づく突然変異を実行する。
        """
        target: str = random.choice(PARAMETER_NAMES)
        before = getattr(chromosome, target)  # 変異前の値を取得
        setattr(chromosome, target, getattr(chromosome, target) + random.gauss(0, 1))
        after = getattr(chromosome, target)  # 変異後の値を取得
        print(f"gaussian_mutation: {target} {before} -> {after}")
    # =================================================================== #
    
    """ Next_Generation """
    def _to_next_generation(self) -> None:
        """
        次世代の個体（染色体）を生成し、個体群の属性値を生成した
        次世代の個体群で置換する。
        """
        new_population: List[Chromosome] = []

        # 元の個体群の件数が奇数件数の場合を加味して件数の比較は等値ではなく
        # 小なりの条件で判定する。
        while len(new_population) < len(self._population):
            parents: List[Chromosome] = self._get_parents_by_selection_type()
            next_generation_chromosomes: List[Chromosome] = \
                self._get_next_generation_chromosomes(parents=parents)
            new_population.extend(next_generation_chromosomes)

        # 2件ずつ次世代のリストを増やしていく都合、元のリストよりも件数が
        # 多い場合は1件リストから取り除いてリストの件数を元のリストと一致させる。
        if len(new_population) > len(self._population):
            del new_population[0]

        self._population = new_population

    def _get_next_generation_chromosomes(
            self, parents: List[Chromosome]) -> List[Chromosome]:
        """
        算出された親の2つの個体のリストから、次世代として扱う
        2つの個体群のリストを取得する。
        一定確率で交叉や変異させ、確率を満たさない場合には引数の値が
        そのまま次世代として設定される。

        Parameters
        ----------
        parents : list of Chromosome
            算出された親の2つの個体のリスト

        Returns
        -------
        next_generation_chromosomes : list of Chromosome
            次世代として設定される、2つの個体を格納したリスト。
        """
        random_val: float = random.random()
        next_generation_chromosomes: List[Chromosome] = parents
        
        if random_val < self._crossover_probability:
            if self._crossover_type == self.CROSSOVER_TYPE_SINGLE_POINT:
                next_generation_chromosomes = self._exec_single_point_crossover(parents)
            elif self._crossover_type == self.CROSSOVER_TYPE_UNIFORM:
                next_generation_chromosomes = self._exec_uniform_crossover(parents)
            else:
                raise ValueError(f"対応していない交叉方式が指定されています: {self._crossover_type}")
        
        random_val: float = random.random()
        if random_val < self._mutation_probability:
            for chromosome in next_generation_chromosomes:
                if self._mutation_type == self.MUTATION_TYPE_UNIFORM:
                    self._exec_uniform_mutation(chromosome)
                elif self._mutation_type == self.MUTATION_TYPE_GAUSSIAN:
                    self._exec_gaussian_mutation(chromosome)
                else:
                    raise ValueError(f"対応していない突然変異方式が指定されています: {self._mutation_type}")
                
        return next_generation_chromosomes

    def _get_parents_by_selection_type(self) -> List[Chromosome]:
        """
        選択方式に応じた親の2つの個体（染色体）のリストを取得する。

        Returns
        -------
        parents : list of Chromosome
            取得された親の2つの個体（染色体）のリスト。

        Raises
        ------
        ValueError
            対応していない選択方式が指定された場合。
        """
        if self._selection_type == self.SELECTION_TYPE_ROULETTE_WHEEL:
            parents: List[Chromosome] = self._exec_roulette_wheel_selection()
        elif self._selection_type == self.SELECTION_TYPE_TOURNAMENT:
            parents = self._exec_tournament_selection()
        elif self._selection_type == self.SELECTION_TYPE_VARIABLE_TOURNAMENT:
            parents = self._exec_variable_tournament_selection()
        else:
            raise ValueError(
                '対応していない選択方式が指定されています : %s'
                % self._selection_type)
        return parents
    # =================================================================== #
    """ 並列化計算のための関数 """
    def _evaluate_population_parallel(self):
        """
        個体群の評価値を並列で計算し、各個体にキャッシュする
        """
        # 各個体のパラメータをdict化
        args_list = []
        for chrom in self._population:
            args = dict(
                parameter_file=chrom.parameter_file,
                parameters=chrom.parameters,
                Bpht=chrom.Bpht,
                Apht=chrom.Apht,
                uu0t=chrom.uu0t,
                so0t=chrom.so0t,
                nt=chrom.nt,
                ndt=chrom.ndt,
                timet=chrom.timet,
                startpoint=chrom.startpoint,
                endpoint=chrom.endpoint,
                Sunspot_N=chrom.Sunspot_N
            )
            args_list.append(args)
        # 並列実行
        with ProcessPoolExecutor() as executor:
            fitness_list = list(executor.map(DefunctionProblem.get_fitness_static, args_list))
        for chrom, fit in zip(self._population, fitness_list):
            chrom._fitness = fit
    
    # =================================================================== #
    """ Run Algorithm """
    def run_algorithm(self) -> Chromosome:
        """
        遺伝的アルゴリズムを実行し、実行結果の個体（染色体）のインスタンス
        を取得する。

        Returns
        -------
        betst_chromosome : Chromosome
            アルゴリズム実行結果の個体。評価関数でしきい値を超えた個体
            もしくはしきい値を超えない場合は指定された世代数に達した
            時点で一番評価関数の値が高い個体が設定される。
        """
        try:
            # 0世代の生成
            self._evaluate_population_parallel()
            best_chromosome: Chromosome = \
                deepcopy(self._get_best_chromosome_from_population())
            
            for generation_idx in range(self._max_generations):
                print(
                    datetime.now(),
                    f'世代数 : {generation_idx}'
                    f' 最良個体情報 : {best_chromosome}'
                )

                if best_chromosome.get_fitness() >= self._threshold:
                    print("=== 閾値到達個体で再シミュレーション ===")
                    args = self._prepare_simulation_args(best_chromosome)
                    result = DefunctionProblem.run_defunction_simulation(**args)
                    print("再シミュレーション結果（相関係数）:", result)
                    return best_chromosome

                self._to_next_generation()
                self._evaluate_population_parallel()
                
                currrent_generation_best_chromosome: Chromosome = \
                    self._get_best_chromosome_from_population()
                current_gen_best_fitness: float = \
                    currrent_generation_best_chromosome.get_fitness()
                if best_chromosome.get_fitness() < current_gen_best_fitness:
                    best_chromosome = deepcopy(currrent_generation_best_chromosome)
            return best_chromosome
        
        except KeyboardInterrupt:
            print("\n=== 実行が中断されました ===")
            print("=== 現時点での最良個体を再計算します ===")
            best_chromosome: Chromosome = \
                deepcopy(self._get_best_chromosome_from_population())
            args = self._prepare_simulation_args(best_chromosome)
            result = DefunctionProblem.run_defunction_simulation(**args)
            print("再シミュレーション結果（相関係数）:", result)
            return best_chromosome
    
    def _prepare_simulation_args(self, chromosome: Chromosome) -> Dict:
        """
        シミュレーション実行用の引数を準備する。

        Parameters
        ----------
        chromosome : Chromosome
            シミュレーションを実行する対象の染色体。

        Returns
        -------
        args : dict
            シミュレーション実行用の引数を格納した辞書。
        """
        return dict(
            parameter_file=chromosome.parameter_file,
            parameters=chromosome.parameters,  # 辞書でパラメータを渡す
            Bpht=chromosome.Bpht,
            Apht=chromosome.Apht,
            uu0t=chromosome.uu0t,
            so0t=chromosome.so0t,
            nt=chromosome.nt,
            ndt=chromosome.ndt,
            timet=chromosome.timet,
            startpoint=chromosome.startpoint,
            endpoint=chromosome.endpoint,
            Sunspot_N=chromosome.Sunspot_N
        )

    def _get_best_chromosome_from_population(self) -> Chromosome:
        """
        個体群のリストから、評価関数の値が一番高い個体（染色体）を
        取得する。

        Returns
        -------
        best_chromosome : Chromosome
            リスト内の評価関数の値が一番高い個体。
        """
        best_chromosome: Chromosome = self._population[0]
        for chromosome in self._population:
            if best_chromosome.get_fitness() < chromosome.get_fitness():
                best_chromosome = chromosome
        return best_chromosome
    # =================================================================== #
class DefunctionProblem(Chromosome):

    def __init__(self, parameters: Dict[str, float], parameter_file, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N) -> None:
        """
        パラメータを辞書で受け取り、インスタンス変数として設定する。
        """
        self.parameters = parameters  # パラメータを辞書で保持
        # ========== #
        """ 以下初期条件にのみ使用"""
        self.parameter_file = parameter_file
        self.Bpht = Bpht
        self.Apht = Apht
        self.uu0t = uu0t
        self.so0t = so0t
        self.nt = nt
        self.ndt = ndt
        self.timet = timet
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.Sunspot_N = Sunspot_N
        
    def get_fitness(self) -> float:
        """
        評価関数として利用するメソッド。

        Returns
        -------
        fitness : float
            相関係数。
        """
        if hasattr(self, '_fitness'):  # すでに計算済みの場合はキャッシュを利用
            return self._fitness
        raise RuntimeError("get_fitnessは並列評価後に呼んでください")
    
    @classmethod
    def make_random_instance(cls, parameter_file, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N) -> DefunctionProblem:
        """
        ランダムな初期値を与えた DefunctionProblem クラスの
        インスタンスを生成する。

        Returns
        -------
        problem : DefunctionProblem
            生成されたインスタンス。xとyには0～99までの範囲でランダムな
            値が設定される。
        """
        import numpy as np
        # TODO: 変更箇所②
        parameters = {
            'a0_s': np.random.uniform(0, 50),
            'a1_s': np.random.uniform(0, 10),
            'a2_s': np.random.uniform(0, 10),
            'a3_s': np.random.uniform(0, 10),
            'b1_s': np.random.uniform(0, 10),
            'b2_s': np.random.uniform(0, 10),
            'b3_s': np.random.uniform(0, 10),
            'omega_s': np.random.uniform(1/(693782000*10), 1/(693782000)),
            'a0_u': np.random.uniform(100, 1300),
            'a1_u': np.random.uniform(0, 300),
            'a2_u': np.random.uniform(0, 300),
            'a3_u': np.random.uniform(0, 300),
            'b1_u': np.random.uniform(0, 300),
            'b2_u': np.random.uniform(0, 300),
            'b3_u': np.random.uniform(0, 300),
            'omega_u': np.random.uniform(1/(693782000*10), 1/(693782000))
        }
        problem = DefunctionProblem(parameters, parameter_file, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N)
        return problem
    

    def __str__(self) -> str:
        """
        個体情報の文字列を返却する。

        Returns
        -------
        info : str
            個体情報の文字列。
        """
        param_str = ', '.join([f'{key} = {value}' for key, value in self.parameters.items()])
        fitness = self.get_fitness()
        return f'{param_str}, fitness = {fitness}'
    
    # パラメタの種類はここで編集
    @staticmethod
    def run_defunction_simulation(parameter_file, parameters, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N):
        """
        実行するシミュレーション
        """
        print(", ".join([f"{key} = {value}" for key, value in parameters.items()]))
        cfg = S2MFD.Cfg(parameter_file)
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        # 両方パターン
        sim.initial_for_defunction(parameters=parameters, Bpht=Bpht, Apht=Apht, uu0t=uu0t, so0t=so0t, nt=nt, ndt=ndt, timet=timet, index=startpoint, index_end=endpoint)
        sim.defunction_main_loop(parameters=parameters, timet=timet, index_start=startpoint, index_end=endpoint)
        
        cc  = sim.judge(Sunspot_N[startpoint:endpoint+1])
        sd  = sim.judge2(Sunspot_N[startpoint:endpoint+1])
        # 評価関数
        alpha = 0.5
        eva = alpha*cc - (1-alpha)*sd
        
        return eva
    @staticmethod
    def get_fitness_static(args):
        """
        並列化用。引数はタプルまたはdictで個体のパラメータを渡す
        """
        # 必要なパラメータを展開
        return DefunctionProblem.run_defunction_simulation(**args)

