from reactpy import component, html, use_state, event, run, widgets, use_effect
from reactpy.backend.flask import configure, Flask
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import pprint as pp
import backend

with open("bootstrap.css") as f:
    css = f.read()

@component
def FormStockTicker(t, set_t, id):
    return html.div(
        {'class': 'mb-3'},
        html.label(
            {
                'class': 'form-label fw-bold',
                'for': 'ticker'
            },
            f"Ticker {id}",
        ),
        html.input(
            {
                'class': 'form-control',
                'type': 'text',
                'id': 'ticker',
                'placeholder': 'Enter Ticker',
                'on_change': lambda event: set_t(event['target']['value']),
                'value': t,
            },
        ),
    )

@component
def FormRiskTolerance(r, set_t):
    return html.div(
        {'class': 'mb-3'},
        html.label(
            {
                'class': 'form-label fw-bold',
                'for': 'risk'
            },
            "Risk Tolerance",
        ),
        html.input(
            {
                'class': 'form-control',
                'type': 'number',
                'id': 'risk',
                'placeholder': 'Enter Risk Tolerance (i.e 0.1 for 10%)',
                'on_change': lambda event: set_t(event['target']['value']),
                'value': r,
            },
        ),
    )

@component
def FormInvestment(i, set_t):
    return html.div(
        {'class': 'mb-3'},
        html.label(
            {
                'class': 'form-label fw-bold',
                'for': 'invest'
            },
            "Investment",
        ),
        html.input(
            {
                'class': 'form-control',
                'type': 'number',
                'id': 'invest',
                'placeholder': 'Enter Investment (i.e 1000 for $1000)',
                'on_change': lambda event: set_t(event['target']['value']),
                'value': i,
            },
        ),
    )

@component
def FormSubmitButton():
    return html.input(
        {
            'class': 'btn btn-primary mb-3',
            'type': 'submit',
            'value': 'Submit'
        },
    )

@component
def Index():
    t1, set_t1 = use_state('')
    t2, set_t2 = use_state('')
    t3, set_t3 = use_state('')
    r, set_r = use_state('')
    i, set_i = use_state('')
    loading, set_loading = use_state("The Model will take 3-5 min to process")
    res_1, set_res1 = use_state("")
    res_2, set_res2 = use_state("")
    res_3, set_res3 = use_state("")

    # Establish Initial Image
    trans_im = Image.open(r"transparent.png")
    trans_im_byte_arr = BytesIO()
    trans_im.save(trans_im_byte_arr, format="PNG")
    image, set_img = use_state(trans_im_byte_arr)

    @use_effect
    async def background_state():
        if loading == "Calculating...":
            print(f"Submitted Tickers {t1}, {t2}, {t3}")
            print(f"Risk Tolerance: {r}")
            print(f"Investment: {i}")
            
            optWeights, optimal_weights_percentages, tickers, investment = backend.submitTickers(t1, t2, t3, r, i)

            # Create Raw results strings
            res_raw1 = f"You should invest ${optWeights[0] * investment:.2f} in {tickers[0].upper()}."
            res_raw2 = f"You should invest ${optWeights[1] * investment:.2f} in {tickers[1].upper()}."
            res_raw3 = f"You should invest ${optWeights[2] * investment:.2f} in {tickers[2].upper()}."
            
            # Set Raw Results
            set_res1(res_raw1)
            set_res2(res_raw2)
            set_res3(res_raw3)

            labels = [f"{tickers[i].upper():<4} = {optimal_weights_percentages[i]:.2f}%" for i in range(len(tickers))]

            plt.clf()
            plt.pie(optimal_weights_percentages, startangle=90)
            plt.axis('equal')
            plt.legend(labels, loc="best")
            plt.title("Optimal Portfolio Allocation")
            buffer = BytesIO()
            plt.savefig(buffer, format='PNG')
            buffer.seek(0)
            set_img(buffer)
            set_loading("Processing Complete!")

    @event(prevent_default=True)
    def submit_tickers(event):
        set_loading("Calculating...")

    return html.article(
        html.style(css),
        html.div(
            {'class': 'container mt-3'},
            html.div(
                {'class': 'row'},
                html.div(
                    {'class': 'col-lg-3'},
                    html.form(
                        {'on_submit': submit_tickers},
                        FormStockTicker(t1, set_t1, 1),
                        FormStockTicker(t2, set_t2, 2),
                        FormStockTicker(t3, set_t3, 3),
                        FormSubmitButton()
                    ),
                    html.div(
                        html.div(
                            html.div(loading),
                            html.div(res_1),
                            html.div(res_2),
                            html.div(res_3)
                        ),
                        widgets.image("png", image.getvalue())
                    )
                ),
                html.div(
                    {'class': 'col-lg-3'},
                    html.form(
                        FormRiskTolerance(r, set_r),
                        FormInvestment(i, set_i),
                    ),
                ),
            ), 
        )
    )

run(Index, port=80)
