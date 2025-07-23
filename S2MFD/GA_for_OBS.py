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
import matplotlib.pyplot as plt
import os
"""
実装項目
=======================================
・初期条件を与えない仕組み：OK
・歴代最良個体の保持：OK
・正解に近い解をフーリエ級数でfittingする仕組み
・観測データインプットの仕組み：OK
  datadirを”OBS”とすると観測データが使われるようにした。datadirにシミュレーション生成したデータを入れると今まで通り。

覚書
=======================================
・Bpht, Apht, uu0t, so0t, nd, ndtは与えないようにしたので、それに伴う変更が要請される。
"""



# TODO: 変更箇所①
# PARAMETER_NAMES = ['a0_u', 'a1_u', 'a2_u', 'b1_u', 'b2_u', 'omega_u', 'u0_const', 's0_const']
# PARAMETER_NAMES = ['a0_s', 'a1_s', 'a2_s', 'b1_s', 'b2_s', 'omega_s', 'u0_const', 's0_const']
# PARAMETER_NAMES = ['a0_u', 'a1_u', 'a2_u', 'omega_u', 'a0_s']
PARAMETER_NAMES = ['a0_s', 'a1_s', 'a2_s', 'omega_s', 'a0_u']
def GA_for_OBS(cfg=None, parameter_file=None, datadir=None, startpoint=0, endpoint=0, output_dir=None, g_num=0):
    """
    観測データのインプット
    """
    start_time = time.time()
    parameter_file = "parameters/" + parameter_file
    if datadir == "OBS":
        data = np.genfromtxt("obs_data/obs_data/SN_Yearly_interp.csv", delimiter=',', skip_header=1)
        timet = data[:, 1]
        timez = data[:, 2]
        Sunspot_N = data[:, 3]
        return
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
        parameter_file, timet, startpoint, endpoint, Sunspot_N, output_dir
    ) for _ in range(g_num)  # 個体数
    ]
    
    """
    GAの設定と実行
    """
    # TODO : 変更箇所②
    ga: GeneticAlgorithm = GeneticAlgorithm(
        initial_population=defunction_initial_population,
        threshold=0.683,
        max_generations=1000,
        mutation_probability=0.3,
        crossover_probability=0.8,
        selection_type=GeneticAlgorithm.SELECTION_TYPE_ASP_TOURNAMENT,  # 選択方式
        crossover_type=GeneticAlgorithm.CROSSOVER_TYPE_SBX,  # 交叉方式
        mutation_type=GeneticAlgorithm.MUTATION_TYPE_GAUSSIAN  # 突然変異方式
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
    SELECTION_TYPE_ASP_TOURNAMENT: SelectionType = 3
    SELECTION_TYPE_ASP_VER2_TOURNAMENT: SelectionType = 4
    
    # 交叉タイプの指定
    CrossoverType = int
    CROSSOVER_TYPE_SINGLE_POINT: CrossoverType = 1
    CROSSOVER_TYPE_UNIFORM: CrossoverType = 2
    CROSSOVER_TYPE_PROT: CrossoverType = 3
    CROSSOVER_TYPE_SBX: CrossoverType = 4  

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
            - SELECTION_TYPE_ASP_TOURNAMENT
        crossover_type : int
            交叉方式。以下のいずれかの定数値を指定する。
            - CROSSOVER_TYPE_SINGLE_POINT
            - CROSSOVER_TYPE_UNIFORM
            - CROSSOVER_TYPE_SBX
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
        1.適応度(fitness)リストの作成
        2.適応度リストを重みとして確率計算しに個体を選択。
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
        Notes
        -----
        全体の半数がトーナメントの参加者となり、評価関数の上位2個体が選択される。この時、半数の抽出は重複を許す。
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
    def _exec_asp_tournament_selection(self) -> List[Chromosome]:
        """
        多様性と適応度変化率に応じて選択圧を調節するトーナメント選択方式。
        
        Parameters
        ----------
        self : GeneticAlgorithm
            
        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。
        """
        print("ASP_TOURNAMENT")
        # 適応度変化率
        if self.generation_idx > 0:
            fitness_delta = abs(self.fitness_story[self.generation_idx] - self.fitness_story[self.generation_idx - 1])/self.fitness_story[self.generation_idx - 1]
        else:
            fitness_delta = 10
        """
        # 最大直近5世代の適応度変化率（差分の絶対値の平均）をとる。最初の方は2,3,4世代の平均を順に取っていく
        if self.generation_idx > 0:
            start_idx = max(0, self.generation_idx - 4)
            recent_deltas = np.abs(np.diff(self.fitness_story[start_idx:self.generation_idx+1]))
            fitness_delta = np.mean(recent_deltas)
        else:
            fitness_delta = 0.0
        """     
        
        # 多様性(直近世代の値)
        diversity = self.diversity()
        
        # トーナメントサイズの決定
        epsi_fit = 0.005 # 適応度変化が0.5%程度しか起きていない→停滞していると判断
        epsi_div = 0.10  # 平均標準偏差が2未満で多様性喪失と判断
        if fitness_delta < epsi_fit:
            if diversity < epsi_div:
                # 適応度変化：低、多様性：低　→ 収束段階だが、局所最適化の可能性を避ける
                # トーナメントサイズ3(選択圧：低)
                participants_num: int = len(self._population) // 10
                participants: List[Chromosome] = choices(self._population, k=participants_num)
                print("選択圧(低)",participants_num,fitness_delta,diversity)
            else:
                # 適応度変化：低、多様性：高　→ 収束段階と判断。緩やかに収束させる。
                # トーナメントサイズ7(選択圧：大)
                participants_num: int = len(self._population) // 4
                participants: List[Chromosome] = choices(self._population, k=participants_num)
                print("選択圧(大)",participants_num,fitness_delta,diversity)
        else:
            # 適応度変化：高　→ 新しい解を探索する段階
            # トーナメントサイズ5(選択圧：中)
            participants_num: int = len(self._population) // 6
            participants: List[Chromosome] = choices(self._population, k=participants_num)
            print("選択圧(中)",participants_num,fitness_delta,diversity)
            
        # ★ fitnessが未計算の場合に例外を防ぐ
        for participant in participants:
            if not hasattr(participant, '_fitness'):
                raise RuntimeError("参加者のfitnessが未計算です。")
            
        # トーナメント参加者から上位2個体を選択
        selected_chromosomes: List[Chromosome] = nlargest(n=2, iterable=participants)
        return selected_chromosomes
    
    def _exec_asp_tournament2_selection(self) -> List[Chromosome]:
        """
        多様性と適応度変化率に応じて選択圧を調節するトーナメント選択方式。
        
        Parameters
        ----------
        self : GeneticAlgorithm
            
        Returns
        -------
        selected_chromosomes : list of Chromosome
            選択された2つの個体（染色体）を格納したリスト。
        """
        print("ASP_TOURNAMENT2")
        Now_fitness = self.fitness_story[self.generation_idx]
        # 適応度変化率
        if self.generation_idx > 0:
            fitness_delta = abs(self.fitness_story[self.generation_idx] - self.fitness_story[self.generation_idx - 1])/self.fitness_story[self.generation_idx - 1]
        else:
            fitness_delta = 10
        """
        # 最大直近5世代の適応度変化率（差分の絶対値の平均）をとる。最初の方は2,3,4世代の平均を順に取っていく
        if self.generation_idx > 0:
            start_idx = max(0, self.generation_idx - 4)
            recent_deltas = np.abs(np.diff(self.fitness_story[start_idx:self.generation_idx+1]))
            fitness_delta = np.mean(recent_deltas)
        else:
            fitness_delta = 0.0
        """     
        
        # 多様性(直近世代の値)
        diversity = self.diversity()
        
        # トーナメントサイズの決定
        epsi_fit = 0.005 # 適応度変化が0.5%程度しか起きていない→停滞していると判断
        epsi_div = 0.10  # 平均標準偏差が2未満で多様性喪失と判断
        if fitness_delta < epsi_fit:
            if  Now_fitness > 0.9*0.881:
                # 適応度変化：低、多様性：高　→ 収束段階と判断。緩やかに収束させる。
                # トーナメントサイズ7(選択圧：大)
                participants_num: int = len(self._population) // 4
                participants: List[Chromosome] = choices(self._population, k=participants_num)
                print("選択圧(大)",participants_num,fitness_delta,diversity)

            else:
                # 適応度変化：低、多様性：低　→ 収束段階だが、局所最適化の可能性を避ける
                # トーナメントサイズ3(選択圧：低)
                participants_num: int = len(self._population) // 10
                participants: List[Chromosome] = choices(self._population, k=participants_num)
                print("選択圧(低)",participants_num,fitness_delta,diversity)
        else:
            # 適応度変化：高　→ 新しい解を探索する段階
            # トーナメントサイズ5(選択圧：中)
            participants_num: int = len(self._population) // 6
            participants: List[Chromosome] = choices(self._population, k=participants_num)
            print("選択圧(中)",participants_num,fitness_delta,diversity)
            
        # ★ fitnessが未計算の場合に例外を防ぐ
        for participant in participants:
            if not hasattr(participant, '_fitness'):
                raise RuntimeError("参加者のfitnessが未計算です。")
            
        # トーナメント参加者から上位2個体を選択
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
    
    def _exec_prot_crossover(self, parents: List[Chromosome]) -> List[Chromosome]:
        """
        プロトタイプ交叉を実行する。
        """
        from copy import deepcopy
        child_1 = deepcopy(parents[0])
        child_2 = deepcopy(parents[1])
        
        child_1.parameters['B'], child_2.parameters['B'] = child_2.parameters['B'], child_1.parameters['B']
        child_1.parameters['C'], child_2.parameters['C'] = child_2.parameters['C'], child_1.parameters['C']
        print(f"prototype_crossover: child_1 {child_1.parameters}, child_2 {child_2.parameters}")
        return [child_1, child_2]
    
    def sbx_crossover(self, parents: List[Chromosome]) -> List[Chromosome]:
        """
        Simulated Binary Crossover (SBX)を実行する。
        """
        from copy import deepcopy
        child_1 = deepcopy(parents[0])
        child_2 = deepcopy(parents[1])
        
        eta_sbx = 2.0
        for attr in PARAMETER_NAMES:
            # 交叉する遺伝子（パラメタ）をランダムに決定
            if random.random() > 0.5:
                u_sbx = random.random()
                if u_sbx <= 0.5:
                    beta_sbx = (2.0 * u_sbx) ** (1.0 / (eta_sbx + 1.0))
                else:
                    beta_sbx = (1.0 / (2.0 * (1.0 - u_sbx))) ** (1.0 / (eta_sbx + 1.0))
                child_1.parameters[attr] = 0.5 * ((1 + beta_sbx) * parents[0].parameters[attr] + (1 - beta_sbx) * parents[1].parameters[attr])
                child_2.parameters[attr] = 0.5 * ((1 - beta_sbx) * parents[0].parameters[attr] + (1 + beta_sbx) * parents[1].parameters[attr])
        print(f"sbx_crossover: child_1 {child_1.parameters}, child_2 {child_2.parameters}")
        return [child_1, child_2]
    
    # =================================================================== #
    # =================================================================== #
    """ def mutation methods """
    def _exec_uniform_mutation(self, chromosome: Chromosome) -> None:
        """
        一様突然変異を実行する。
        """
        target: str = random.choice(PARAMETER_NAMES)
        before = chromosome.parameters[target]
        new_value = before * random.uniform(0.9, 1.1)
        chromosome.parameters[target] = new_value 
        after = chromosome.parameters[target]
        print(f"uniform_mutation: {target} {before} -> {after}")

    def _exec_gaussian_mutation(self, chromosome: Chromosome) -> None:
        """
        ガウス分布に基づく突然変異を実行する。
        """
        target: str = random.choice(PARAMETER_NAMES)
        before = chromosome.parameters[target]
        sigma = 0.1 * abs(before)  # 標準偏差は値の10%（必要に応じて調整）
        new_value = np.random.normal(loc=before, scale=sigma)
        chromosome.parameters[target] = new_value 
        after = chromosome.parameters[target] # 変異後の値を取得
        print(f"gaussian_mutation: {target} {before} -> {after}")
    # =================================================================== #
    
    """ Next_Generation """
    def _to_next_generation(self) -> None:
        """
        次世代の個体（染色体）を生成し、個体群の属性値を生成した
        次世代の個体群で置換する。
        """
        new_population: List[Chromosome] = []

        # 新世代個体群の生成
        while len(new_population) < len(self._population):
            parents: List[Chromosome] = self._get_parents_by_selection_type()
            next_generation_chromosomes: List[Chromosome] = \
                self._get_next_generation_chromosomes(parents=parents)
            new_population.extend(next_generation_chromosomes)

        # 現世代の最良個体（エリート）を取得
        new_population[3] = deepcopy(self.best_chromosome)
        print(self.best_chromosome)

        # 個体数調整（新世代が多い場合は削除）
        if len(new_population) > len(self._population):
            del new_population[0]
        
        # 個体群の置換
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
            elif self._crossover_type == self.CROSSOVER_TYPE_PROT:
                next_generation_chromosomes = self._exec_prot_crossover(parents)
            elif self._crossover_type == self.CROSSOVER_TYPE_SBX:
                next_generation_chromosomes = self.sbx_crossover(parents)
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
        elif self._selection_type == self.SELECTION_TYPE_ASP_TOURNAMENT:
            parents = self._exec_asp_tournament_selection()
        elif self._selection_type == self.SELECTION_TYPE_ASP_VER2_TOURNAMENT:
            parents = self._exec_asp_tournament2_selection()
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
                timet=chrom.timet,
                startpoint=chrom.startpoint,
                endpoint=chrom.endpoint,
                Sunspot_N=chrom.Sunspot_N,
                output_dir= chrom.output_dir
            )
            args_list.append(args)
        # 並列実行
        with ProcessPoolExecutor() as executor:
            fitness_list = list(executor.map(DefunctionProblem.get_fitness_static, args_list))
        for chrom, fit in zip(self._population, fitness_list):
            chrom._fitness = fit
    
    # =================================================================== #
    # ================================================================== #
    """描画用関数"""
    def draw_population(self,x,y,label,label_x,label_y,file_name):
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
    # =================================================================== #
    # ================================================================== #
    """"多様性を図る関数"""
    def diversity(self) -> float:
        """
        現在の世代の多様性を計算する。

        Returns
        -------
        diversity_score : float
            個体群の多様性を表すスコア（標準偏差の平均値）。
        """
        # パラメータ値を格納する辞書を初期化
        parameter_values = {key: [] for key in PARAMETER_NAMES}
        
        # 各個体のパラメータ値を収集（パラメータごとに集めている）
        for chromosome in self._population:
            for key in PARAMETER_NAMES:
                parameter_values[key].append(chromosome.parameters[key])
                # print(parameter_values)
                
        # 各パラメータの正規化を実行
        normalized_values = {}
        for key, values in parameter_values.items():
            min_val = np.min(values)  # 最小値
            max_val = np.max(values)  # 最大値
            if max_val - min_val == 0:
                # 値が一定の場合はそのまま使用（正規化の分母=0となってしまうため）
                normalized_values[key] = values
            else:
                # 正規化: (値 - 最小値) / (最大値 - 最小値)
                normalized_values[key] = [(val - min_val) / (max_val - min_val) for val in values]

        # 各パラメータの標準偏差を計算（標準偏差のリストを生成）
        std_devs = [np.std(values) for values in normalized_values.values()]

        # 標準偏差の平均を多様性スコアとして返す
        diversity_score = np.mean(std_devs)
        return diversity_score
        
    
    # ================================================================== #
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
            chrom = self._population[0]
            cfg = S2MFD.Cfg(chrom.parameter_file)
            cfg.datadir = chrom.output_dir
            sim = S2MFD.Simulation(cfg)
            sim.initialize_simulation()
            
            self._evaluate_population_parallel()
            best_chromosome: Chromosome = \
                deepcopy(self._get_best_chromosome_from_population())
            fitness_story   = np.zeros(self._max_generations, dtype=float)
            diversity_story = np.zeros(self._max_generations, dtype=float)
            for generation_idx in range(self._max_generations):
                print(
                    datetime.now(),
                    f'世代数 : {generation_idx}'
                    f' 最良個体情報 : {best_chromosome}'
                )
                fitness_story[generation_idx]   = best_chromosome.get_fitness()
                diversity_story[generation_idx] = self.diversity()
                # ASP使用
                self.fitness_story = fitness_story
                self.generation_idx = generation_idx
                self.best_chromosome = best_chromosome
                
                print("fitness:", fitness_story)
                print("diversity:", diversity_story)

                if best_chromosome.get_fitness() >= self._threshold:
                    print("=== 閾値到達個体で再シミュレーション ===")
                    args = self._prepare_simulation_args(best_chromosome)
                    result = DefunctionProblem.run_last_simulation(**args)
                    print("再シミュレーション結果（相関係数）:", result)
                    
                    self.draw_population(
                        x=np.linspace(0,len(fitness_story[:generation_idx+1])-1,len(fitness_story[:generation_idx+1])),
                        y=fitness_story[:generation_idx+1],
                        label="Fitness",
                        label_x="generation number",
                        label_y="fitness",
                        file_name="fitness_time.png"
                    )
                    self.draw_population(
                        x=np.linspace(0, len(diversity_story[:generation_idx+1]) - 1, len(diversity_story[:generation_idx+1])),
                        y=diversity_story[:generation_idx+1],
                        label="Diversity",
                        label_x="generation number",
                        label_y="diversity",
                        file_name="diversity_time.png"
                    )
                    
                    return best_chromosome

                self._to_next_generation()
                self._evaluate_population_parallel()
                
                currrent_generation_best_chromosome: Chromosome = \
                    self._get_best_chromosome_from_population()
                current_gen_best_fitness: float = \
                    currrent_generation_best_chromosome.get_fitness()
                    
                # 今回の変異で歴代新記録を出した場合に更新する。
                if best_chromosome.get_fitness() < current_gen_best_fitness:
                    best_chromosome = deepcopy(currrent_generation_best_chromosome)
            return best_chromosome
        
        except KeyboardInterrupt:
            self.draw_population(
                x=np.linspace(0,len(fitness_story[:generation_idx+1])-1,len(fitness_story[:generation_idx+1])),
                y=fitness_story[:generation_idx+1],
                label="Fitness",
                label_x="generation number",
                label_y="fitness",
                file_name="fitness_time.png"
            )
            self.draw_population(
                x=np.linspace(0, len(diversity_story[:generation_idx+1]) - 1, len(diversity_story[:generation_idx+1])),
                y=diversity_story[:generation_idx+1],
                label="Diversity",
                label_x="generation number",
                label_y="diversity",
                file_name="diversity_time.png"
            )
            print("\n=== 実行が中断されました ===")
            print("=== 現時点での最良個体を再計算します ===")
            args = self._prepare_simulation_args(best_chromosome)
            result = DefunctionProblem.run_last_simulation(**args)
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
            timet=chromosome.timet,
            startpoint=chromosome.startpoint,
            endpoint=chromosome.endpoint,
            Sunspot_N=chromosome.Sunspot_N,
            output_dir=chromosome.output_dir
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

    def __init__(self, parameters: Dict[str, float], parameter_file, timet, startpoint, endpoint, Sunspot_N, output_dir) -> None:
        """
        パラメータを辞書で受け取り、インスタンス変数として設定する。
        """
        self.parameters = parameters  # パラメータを辞書で保持
        # ========== #
        """ 以下初期条件にのみ使用"""
        self.parameter_file = parameter_file
        self.timet = timet
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.Sunspot_N = Sunspot_N
        self.output_dir = output_dir
        
    def get_fitness(self) -> float:
        """
        評価関数として利用するメソッド。

        Returns
        -------
        fitness : float
        
        Notes
        -----
        ここではfitnessの呼び出しを行うが、実際の計算は並列評価後に行われるため、キャッシュされた値を返す。
        """
        if hasattr(self, '_fitness'):  # すでに計算済みの場合はキャッシュを利用
            return self._fitness
        raise RuntimeError("get_fitnessは並列評価後に呼んでください")
    
    @classmethod
    def make_random_instance(cls, parameter_file, timet, startpoint, endpoint, Sunspot_N, output_dir) -> DefunctionProblem:
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
        # TODO: 変更箇所③
        parameters = {
            'a0_s': np.random.uniform(40, 65),
            'a1_s': np.random.uniform(-15, 15),
            'a2_s': np.random.uniform(-15, 15),
            # 'b1_s': np.random.uniform(-15, 15),
            # 'b2_s': np.random.uniform(-15, 15),
            'omega_s': np.random.uniform(2*np.pi/(30*365*60*60*100), 2*np.pi/(10*365*60*60*100)),
            'a0_u': np.random.uniform(800, 1300)
            # 'a1_u': np.random.uniform(-150, 150),
            # 'a2_u': np.random.uniform(-150, 150),
            # 'b1_u': np.random.uniform(-150, 150),
            # 'b2_u': np.random.uniform(-150, 150),
            # 'omega_u': np.random.uniform(2*np.pi/(30*365*60*60*100), 2*np.pi/(10*365*60*60*100)),
            # 'a0_s': np.random.uniform(40, 65)

        }
        problem = DefunctionProblem(parameters, parameter_file, timet, startpoint, endpoint, Sunspot_N, output_dir)
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
    
    @staticmethod
    def run_defunction_simulation(parameter_file, parameters, timet, startpoint, endpoint, Sunspot_N, output_dir):
        """
        実行するシミュレーション
        """
        print(", ".join([f"{key} = {value}" for key, value in parameters.items()]))
        cfg = S2MFD.Cfg(parameter_file)
        
        cfg.datadir = output_dir
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_for_OBS(parameters=parameters,timet=timet, index_start=startpoint, index_end=endpoint)
        sim.defunction_main_loop(parameters=parameters, timet=timet, index_start=startpoint, index_end=endpoint)
        
        cc  = sim.judge(Sunspot_N[startpoint:endpoint+1])
        sd  = sim.judge2(Sunspot_N[startpoint:endpoint+1])
        
        # TODO 変更箇所④     
        alpha = 0.7
        eva = alpha*cc - (1-alpha)*sd
        
        return eva
    @staticmethod
    def run_last_simulation(parameter_file, parameters, timet, startpoint, endpoint, Sunspot_N, output_dir):
        """
        実行するシミュレーション
        """
        print(", ".join([f"{key} = {value}" for key, value in parameters.items()]))
        cfg = S2MFD.Cfg(parameter_file)
        
        cfg.datadir = output_dir
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_for_LAST(parameters=parameters,timet=timet, index_start=startpoint, index_end=endpoint)
        sim.defunction_main_loop(parameters=parameters, timet=timet, index_start=startpoint, index_end=endpoint)
        
        cc  = sim.judge(Sunspot_N[startpoint:endpoint+1])
        sd  = sim.judge2(Sunspot_N[startpoint:endpoint+1])
        
        # TODO 変更箇所⑤     
        alpha = 0.7
        eva = alpha*cc - (1-alpha)*sd
        
        return eva
    @staticmethod
    def get_fitness_static(args):
        """
        並列化用。引数はタプルまたはdictで個体のパラメータを渡す
        """
        # 必要なパラメータを展開
        return DefunctionProblem.run_defunction_simulation(**args)

