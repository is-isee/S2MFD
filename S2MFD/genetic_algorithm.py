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

def GA_defunction(cfg=None, parameter_file=None, datadir=None, startpoint=0, endpoint=0):
    """
    観測データのインプット
    """
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
    ) for _ in range(30)
    ]
    
    """
    GAの設定と実行
    """
    ga: GeneticAlgorithm = GeneticAlgorithm(
        initial_population=defunction_initial_population,
        threshold=0.99,
        max_generations=1000,
        mutation_probability=0.2,
        crossover_probability=0.5,
        selection_type=GeneticAlgorithm.SELECTION_TYPE_TOURNAMENT)
    _ = ga.run_algorithm()
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

    @abstractmethod
    def mutate(self) -> None:
        """
        染色体を突然変異させる処理の抽象メソッド。
        インスタンスの属性などのランダムな別値の設定などが実行される。
        """
        ...

    @abstractmethod
    def exec_crossover(self, other: Chromosome) -> List[Chromosome]:
        """
        引数に指定された別の個体を参照し交叉を実行する。

        Parameters
        ----------
        other : Chromosome
            交叉で利用する別の個体。

        Returns
        -------
        result_chromosomes : list of Chromosome
            交叉実行後に生成された2つの個体（染色体）。
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
        # TODO
        return self.get_fitness() < other.get_fitness()
C = TypeVar('C', bound=Chromosome)
class GeneticAlgorithm:
    # 選択タイプの指定
    SelectionType = int
    SELECTION_TYPE_ROULETTE_WHEEL: SelectionType = 1
    SELECTION_TYPE_TOURNAMENT: SelectionType = 2

    def __init__(
            self, initial_population: List[C],
            threshold: float,
            max_generations: int, mutation_probability: float,
            crossover_probability: float,
            selection_type: SelectionType) -> None:
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
        """
        self._population: List[Chromosome] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_probability: float = mutation_probability
        self._crossover_probability: float = crossover_probability
        self._selection_type: int = selection_type
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
    # =================================================================== #
    
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
            next_generation_chromosomes = parents[0].exec_crossover(
                other=parents[1])

        random_val = random.random()
        if random_val < self._mutation_probability:
            for chromosome in next_generation_chromosomes:
                chromosome.mutate()
        # for chromosome in next_generation_chromosomes:
        #     if hasattr(chromosome, '_fitness'):
        #         del chromosome._fitness
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
                A_sample=chrom.A,
                omg_sample=chrom.Omg,
                B_sample=chrom.B,
                C_sample=chrom.C,
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
                args = dict(
                    parameter_file=best_chromosome.parameter_file,
                    A_sample=best_chromosome.A,
                    omg_sample=best_chromosome.Omg,
                    B_sample=best_chromosome.B,
                    C_sample=best_chromosome.C,
                    Bpht=best_chromosome.Bpht,
                    Apht=best_chromosome.Apht,
                    uu0t=best_chromosome.uu0t,
                    so0t=best_chromosome.so0t,
                    nt=best_chromosome.nt,
                    ndt=best_chromosome.ndt,
                    timet=best_chromosome.timet,
                    startpoint=best_chromosome.startpoint,
                    endpoint=best_chromosome.endpoint,
                    Sunspot_N=best_chromosome.Sunspot_N
                )
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

    def __init__(self, A: float, Omg: float, B: float, C: float,
                 parameter_file, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N) -> None:
        """
        sin関数のパラメタを定義する。

        Parameters
        ----------
        A : float
            sin関数の振幅。
        Omg : float
            sin関数の角周波数。
        B : float
            sin関数の切片。
        C : float
            sin関数の初期位相。
        """
        self.A = A
        self.Omg = Omg
        self.B = B
        self.C = C
        
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
        print("A=",self.A,"ω=", self.Omg, "B=",self.B, "C=",self.C,"パラメタファイル(確認用)",self.parameter_file)
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
        A:   float = random.uniform(60,100)
        Omg: float = random.uniform(1/7e8, 1/6.5e8)
        B:   float = random.uniform(300,1000)
        C:   float = random.uniform(0,2*np.pi)
        problem = DefunctionProblem(A, Omg, B, C, parameter_file, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N)
        return problem
    
    def mutate(self) -> None:
        """
        ランダムに一つの変数を選択し、変異させる。
        """
        # ランダムに変異させるターゲットを決定
        target: str = choices(['A', 'Omg', 'B', 'C'], k=1)[0]
        before = getattr(self, target)
        # 5%増減させる
        if random.random() > 0.5:
            setattr(self, target, getattr(self, target) * 1.05)
        else:
            setattr(self, target, getattr(self, target) * 0.95)
        after = getattr(self, target)
        print(f"mutate: {target} {before} -> {after}")
        # if hasattr(self, '_fitness'):
        #     del self._fitness  # キャッシュ削除

    def exec_crossover(
            self, other: DefunctionProblem
            ) -> List[DefunctionProblem]:
        """
        引数に指定された別の個体を参照し交叉を実行する。

        Parameters
        ----------
        other :DefunctionProblem
            交叉で利用する別の個体。

        Returns
        -------
        result_chromosomes : list of DefunctionProblem
            交叉実行後に生成された2つの個体を格納したリスト。親となる
            個体それぞれから、半分ずつ受け継いだ個体となる。
        """
        from copy import deepcopy
        child_1 = deepcopy(self)
        child_2 = deepcopy(other)
        child_1.B = other.B
        child_1.C = other.C
        child_2.B = self.B
        child_2.C = self.C
        # if hasattr(child_1, '_fitness'):
        #     del child_1._fitness
        # if hasattr(child_2, '_fitness'):
        #     del child_2._fitness
        print(f"crossover: child_1 {[getattr(child_1, a) for a in ['A','Omg','B','C']]}, child_2 {[getattr(child_2, a) for a in ['A','Omg','B','C']]}")
        return [child_1, child_2]        

    def __str__(self) -> str:
        """
        個体情報の文字列を返却する。

        Returns
        -------
        info : str
            個体情報の文字列。
        """
        A:   float = self.A
        Omg: float = self.Omg
        B:   float = self.B
        C:   float = self.C
        fitness: float = self.get_fitness()
        info: str = f'A = {A}, Omg = {Omg}, B = {B}, C = {C}, fitness = {fitness}'
        return info
    @staticmethod
    def run_defunction_simulation(parameter_file, A_sample, omg_sample, B_sample, C_sample, Bpht, Apht, uu0t, so0t, nt, ndt, timet, startpoint, endpoint, Sunspot_N):
        """
        実行するシミュレーション
        """
        print("A=", A_sample, "ω=", omg_sample, "B=", B_sample, "C=", C_sample, "パラメタファイル(確認用)", parameter_file)
        cfg = S2MFD.Cfg(parameter_file)
        sim = S2MFD.Simulation(cfg)
        sim.initialize_simulation()
        sim.cfl_condition()
        sim.initial_for_defunction(A_sample=A_sample, omg_sample=omg_sample, B_sample=B_sample, C_sample=C_sample, Bpht=Bpht, Apht=Apht, uu0t=uu0t, so0t=so0t, nt=nt, ndt=ndt, timet=timet, index=startpoint, index_end=endpoint)
        sim.defunction_main_loop(A_sample=A_sample, omg_sample=omg_sample, B_sample=B_sample, C_sample=C_sample, timet=timet, index_start=startpoint, index_end=endpoint)
        cc = sim.judge(Sunspot_N[startpoint:endpoint+1])
        
        return cc
    @staticmethod
    def get_fitness_static(args):
        """
        並列化用。引数はタプルまたはdictで個体のパラメータを渡す
        """
        # 必要なパラメータを展開
        return DefunctionProblem.run_defunction_simulation(**args)

