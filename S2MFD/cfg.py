import numpy as np
import json, os
import importlib

# 派生パラメタの定義: (名前, 依存する基本量, 計算式)
# 計算式は parameters/defaults.py の定義と同一の式であること(ビット一致が前提)。
# 順序に意味がある(ome → omc, c2 の順に評価)。
_DERIVATIONS = [
    ('ome', ('com', 'RSUN', 'ett'), lambda com, RSUN, ett: com / RSUN**2 * ett),
    ('omc', ('ome',), lambda ome: 0.92 * ome),
    ('c2', ('ome',), lambda ome: 0.2 * ome),
    ('so0', ('cso', 'ett', 'RSUN'), lambda cso, ett, RSUN: cso * ett / RSUN),
    ('uu0', ('rey', 'ett', 'RSUN'), lambda rey, ett, RSUN: rey * ett / RSUN),
    # hotta10 系 (D99/H10 子午面流)
    ('xi0', ('RSUN', 'rrb'), lambda RSUN, rrb: RSUN / rrb - 1),
    ('c1d', ('m', 'p', 'xi0'),
     lambda m, p, xi0: (2 * m + 1) * (m + p) / (m + 1) / p * (xi0**(-m))),
    ('c2d', ('m', 'p', 'xi0'),
     lambda m, p, xi0: (2 * m + p + 1) * m / (m + 1) / p * (xi0**(-(m + p)))),
]


class Cfg:
    """
    Class for configuration management.

    """

    def __init__(self, parameter_file='parameters/defaults.py'):
        """
        Parameters
        ----------
            parameter_file : str
                file path to the parameter file.
                If the path exists as given (absolute or relative to the
                current directory), it is used as is. Otherwise it is
                resolved relative to the S2MFD package directory.

        Notes
        -----
        The parameter file should be a Python file with the following structure:

        .. code-block:: python

            from S2MFD.parameters.defaults import *
            #(change parameters from defaults.py here)
        """
        if not os.path.exists(parameter_file):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            parameter_file = os.path.join(base_dir, parameter_file)

        self.parameter_file = parameter_file
        # Load the parameter file
        self.load_parameters(parameter_file)

    def load_parameters(self, parameter_file):
        '''
        Load parameters from an external Python file.

        Parameters
        ----------
            parameter_file : str
                file path to the parameter file.

        '''
        # Load the parameter_file as a module
        spec = importlib.util.spec_from_file_location("parameters", parameter_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Set self attributes for all items in the module
        for k, v in vars(module).items():
            if not k.startswith("__"):  # Skip special attributes
                setattr(self, k, v)

        self._init_derivations()

    def _init_derivations(self):
        """派生パラメタの追跡を初期化する。

        パラメタファイルが標準の導出式と異なる値を明示的に与えている場合
        (例: hotta10.py の ome, uu0)、その名前は「固定 (pinned)」扱いとなり
        resolve() でも再計算しない。
        """
        self._pinned = set()
        self._derived_cache = {}
        for name, inputs, fn in _DERIVATIONS:
            if not all(hasattr(self, i) for i in inputs):
                continue
            if not hasattr(self, name):
                continue
            expected = fn(*[getattr(self, i) for i in inputs])
            if getattr(self, name) == expected:
                self._derived_cache[name] = expected
            else:
                self._pinned.add(name)

    def resolve(self):
        """基本量の変更を派生パラメタに反映する。

        例: ``cfg.rey = 1400; cfg.resolve()`` とすると uu0 が再計算される。
        Simulation の初期化時にも自動で呼ばれる。

        再計算しないもの:

        - パラメタファイルが導出式と異なる値を明示していた名前 (pinned)
        - 前回の resolve()/読込以降にユーザーが直接値を代入した名前
          (例: ``cfg.uu0 = 500`` とした場合、rey からの再計算で上書きしない)
        """
        if not hasattr(self, '_pinned'):
            self._init_derivations()
        for name, inputs, fn in _DERIVATIONS:
            if name in self._pinned:
                continue
            if not all(hasattr(self, i) for i in inputs):
                continue
            current = getattr(self, name, None)
            if name in self._derived_cache and current is not None \
                    and not _values_equal(current, self._derived_cache[name]):
                # ユーザーが直接代入した → 以後は固定
                self._pinned.add(name)
                continue
            value = fn(*[getattr(self, i) for i in inputs])
            setattr(self, name, value)
            self._derived_cache[name] = value
        return self

    def save(self):
        """
        Save configuration to a JSON file.
        The data is store in self.configfile defined in the parameter file.
        """
        params = {}
        for k in dir(self):
            if k.startswith("_") or callable(getattr(self, k)):
                continue
            value = _to_native(getattr(self, k))
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                params[k] = value
        with open(os.path.join(self.datadir, self.configfile), 'w') as f:
            json.dump(params, f, indent=4)

    @classmethod
    def load(cls, filename):
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        filename : str
            file path to the JSON file.
        """
        obj = cls.__new__(cls)
        with open(filename, 'r') as f:
            params = json.load(f)
        for k, v in params.items():
            setattr(obj, k, v)

        return obj


def _to_native(value):
    """numpy スカラー等を JSON 保存可能な Python ネイティブ型に変換する。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _values_equal(a, b):
    try:
        return bool(a == b)
    except Exception:
        return False


def build_cfg(parameter_file='parameters/defaults.py', datadir=None,
              resolve=True, **overrides):
    """パラメタファイルから :class:`Cfg` を作り, 必要なら値を上書きする.

    設定を作る入口をここ 1 つにするための関数。実行スクリプトも解析
    スクリプトもテストもこれを使う。

    Parameters
    ----------
    parameter_file : str
        パラメタファイル。``'parameters/xxx.py'`` のように書くと
        パッケージ同梱のものを読む (``S2MFD/parameters/`` 以下)。
        絶対パスを渡せば任意のファイルも読める。
    datadir : str or pathlib.Path, optional
        出力先。末尾のスラッシュは自動で補う (現実装はパスを文字列
        連結するため必須)。
    resolve : bool
        ``True`` (既定) なら上書きのあとに :meth:`Cfg.resolve` を呼び、
        **基本量の変更を派生量に反映する**。

        これを呼ばないと ``build_cfg(..., rey=1400)`` としても
        ``uu0 = rey*ett/RSUN`` が古いままになる (実測: ``uu0`` が 0 の
        まま変わらない)。:class:`~S2MFD.Simulation` の初期化では自動で
        呼ばれるが、動力学モードの実行スクリプトは ``Simulation`` を
        作らないので、ここで呼ぶ必要がある。

        パラメタファイルが導出式と**違う値を明示していた**名前と、
        利用者が直接代入した名前は再計算されない
        (:meth:`Cfg.resolve` を参照)。
    **overrides
        ``cfg`` の属性を上書きする。

    Returns
    -------
    Cfg

    Examples
    --------
    論文設定を 216x144 で:

    >>> import S2MFD
    >>> cfg = S2MFD.build_cfg('parameters/rempel06_paper.py', ix=216, jx=144)
    >>> grid = S2MFD.Grid.from_cfg(cfg)

    基本量を変えると派生量も追随する:

    >>> cfg = S2MFD.build_cfg('parameters/alpha_omega.py', rey=1400)
    >>> round(cfg.uu0)          # uu0 = rey*ett/RSUN
    2011
    """
    cfg = Cfg(parameter_file)
    if datadir is not None:
        cfg.datadir = str(datadir).rstrip('/') + '/'
    for key, value in overrides.items():
        setattr(cfg, key, value)
    if resolve:
        cfg.resolve()
    return cfg
