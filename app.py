from functools import partial
from pathlib import Path

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from shiny import reactive, render, ui, App, Inputs, Outputs, Session

from adas import ADAS, AdasParam
from util import ui_katex, ui_css, ui_card_header

card_header = partial(ui.card_header, class_="bg-dark") 


page1 = ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    ui.card(
                        ui.card_header("パラメータ"),
                        ui.card_body(
                            ui.input_numeric("alpha", r"$\alpha$", 1.0, min=0.0, step=0.1),
                            ui.input_numeric("phi", r"$\varphi$", 0.25, min=0.0, step=0.05),
                            ui.input_numeric("theta_pi", r"$\theta_\pi$", 0.5, min=-0.5, step=0.05),
                            ui.input_numeric("theta_Y", r"$\theta_Y$", 0.5, min=-0.5, step=0.05),
                            ui.output_ui("eigvals"),
                            class_="floating-label",
                        ),
                    ),
                    ui.card(
                        ui.card_header("外生変数の定常状態値（初期値の決定に用いる）"),
                        ui.card_body(
                            ui.input_numeric("pi_star", r"$\pi^*$", 2.0, min=0.0, step=0.05),
                            ui.input_numeric("Y_bar", r"$\bar{Y}$", 100., min=0.0, step=1.0),
                            ui.input_numeric("rho", r"$\rho$", 2.0, min=0.0, step=0.05),
                            class_="floating-label",
                        ),
                    )
                ),
                width="20%",
            ),
            ui.layout_columns(        
                ui.div(
                    ui.card(
                        ui_card_header("シミュレーション設定"),
                        ui.input_select("preset", "プリセット", 
                                        {"free": "―",
                                        "high_inflation": "1期限りの供給ショック（インフレショック）",
                                        "high_demand": "5期続く需要ショック",
                                        "lower_target": "目標インフレ率の下落"}),

                        ui.hr(),

                        ui.input_select("senario", "シナリオ", 
                                        {"inflation": "インフレショック", 
                                        "demand": "需要ショック", 
                                        "target": "インフレ目標", 
                                        "potential": "潜在GDP", 
                                        "naturalrate": "自然利子率"}),
                        ui.input_numeric("T", "シミュレーションの長さ", 25, min=5),
                        ui.layout_column_wrap(
                            ui.input_numeric("begin", "ショックの開始期", 1, min=1),
                            ui.div(
                                ui.input_numeric("end", "ショックの終了期", 1, min=1),
                                ui.output_ui("b_lt_e"),
                            ),
                        ),
                        ui.input_numeric("size", "ショックの大きさ", 1),
                    ),
                ),
                ui.card(
                    ui_card_header("シミュレーション結果"),
                    ui.output_plot("plot_area"),
                ),
                col_widths=(4, 8),
            ),
        )

app_ui = ui.page_navbar(
    ui.nav_panel("シミュレーション", page1),
    ui.nav_control(
        ui.a("README", href="https://github.com/kenjisato/dynamic-adas/blob/main/README.md", target="_blank")
    ),
    ui_katex,
    ui_css,
)

def server(input: Inputs, output: Outputs, session: Session):

    # 内生変数
    endog = (("pi", r"$\pi$"),
             ("Epi", r"$E\pi$"),
             ("Y", "$Y$"),
             ("i", "$i$"),
             ("r", "$r$"))

    # 外生変数       
    exog = (("nu", r"$\nu$"),
            ("eps", r"$\varepsilon$"),
            ("pistar", r"$\Delta \pi^*$"),
            ("Ybar", r"$\Delta \bar Y$"),
            ("rho", r"$\Delta \rho$"))
    

    # シミュレーション設定画変更されたとき、プリセット（定義済みのシナリオ）を解除
    @reactive.effect
    @reactive.event(input.begin, input.end, input.size, input.senario)
    def reset_preset():
        ui.update_select("preset", selected="free")

    # プリセット（定義済みのシナリオ）を選択したときの動作
    # プリセットの設定はマンキューの教科書に従った
    @reactive.effect
    @reactive.event(input.preset)
    def preset():

        match input.preset():
            case "high_inflation":
                begin, end, senario, size = 1, 1, "inflation", 1
   
            case "high_demand":
                begin, end, senario, size = 1, 5, "demand", 1
            
            case "lower_target":
                begin, end, senario, size = 3, input.T(), "target", -1
            
            # "ー" を選んだ場合は何もしない
            case "free":
                return None
            
        ui.update_numeric("begin", value=begin)
        ui.update_numeric("end", value=end)
        ui.update_select("senario", selected=senario)
        ui.update_numeric("size", value=size)


    # 開始 > 終了の場合にだけ警告メッセージを表示する     
    @render.ui
    def b_lt_e():
        if input.begin() > input.end():
            return ui.div(ui.help_text("開始 ≦ 終了 が必要です！"), class_="suggestion")
        else:
            return None

    # 入力されたパラメータからAD-ASモデルのクラスを構築する
    @reactive.calc
    def model():
        param = AdasParam(
            alpha = input.alpha(),
            phi = input.phi(),
            theta_pi = input.theta_pi(),
            theta_Y = input.theta_Y(),
            pi_star = input.pi_star(),
            Y_bar = input.Y_bar(),
            rho = input.rho()
        )
        return ADAS(param)
    
    # 係数行列の固有値を計算して、安定性に関するメッセージを表示する
    @render.ui
    def eigvals():
        
        A = model().A
        E, V = LA.eig(A)

        maxeig = np.max(np.abs(E))
        msg = f"固有値の絶対値の最大は約 {maxeig:.3f} です。"

        if maxeig < 1.0:
            msg = msg + "動学システムは安定です。"
        else:
            msg = msg + "動学システムは不安定です。"

        return ui.help_text(msg) 


    # 入力されたシナリオをもとにショックの時系列を構築する
    @reactive.calc
    def shock():

        e = np.zeros(input.T() + 1)

        match input.senario():
            case "inflation":
                i = 0
                diff = input.size() 
            case "demand":
                i = 1
                diff = input.size()
            case "target":
                i = 2
                diff = input.size()
            case "potential":
                i = 3
                diff = input.size()
            case "naturalrate":
                i = 4
                diff = input.size()

        e[(input.begin()):(input.end()+1)] = e[(input.begin()):(input.end()+1)] + diff

        return (i, e)


    # shock() で作られた時系列をもとに、外生変数の時系列 u を構築する
    @reactive.calc
    def u():
        param = model().param

        u = np.zeros((input.T() + 1, len(exog)))     # 1列目がインフレショック、2列目が需要ショック
        u[:, 2] = param.pi_star
        u[:, 3] = param.Y_bar
        u[:, 4] = param.rho
        i, e = shock()
        
        u[:, i] = u[:, i] + e

        return u
        
    # 内生変数の時系列を計算する
    @reactive.calc
    def simulate():
        x0 = model().x0([0., 0., model().param.pi_star, model().param.Y_bar, model().param.rho])
        return model().simulate(u(), x0)

    # 内生変数の時系列をプロットする
    @output
    @render.plot
    def plot_area():
        fig, axes = plt.subplots(5, 1)

        try: 
            x = simulate()

        # A が逆行列を持たないときの処理
        except LA.LinAlgError:
            return fig
        
        i, diff = shock()

        # Shock
        axes[0].plot(diff, marker='.')
        axes[0].set_ylabel(exog[i][1])
             
        cnt = 1
        for i, (key, val) in enumerate(endog):

            # 期待インフレ率は表示しない（このモデルでは π と同じ）
            if key == "Epi":
                continue

            axes[cnt].plot(x[:, i], marker='.')
            axes[cnt].set_ylabel(val)
            cnt += 1

        # 横軸の tick を整数値に強制する
        for i in range(5):
            axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))


www = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=www)


