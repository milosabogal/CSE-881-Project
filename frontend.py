
from reactpy import component, html, hooks, event, utils, run
from reactpy.backend.flask import configure, Flask
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
                'placeholder': 'Enter Risk Tolerance',
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
                'placeholder': 'Enter Investment Amount',
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
    t1, set_t1 = hooks.use_state('')
    t2, set_t2 = hooks.use_state('')
    t3, set_t3 = hooks.use_state('')
    r, set_r = hooks.use_state(0)
    i, set_i = hooks.use_state(0)
    res_1, set_res1 = hooks.use_state("")
    res_2, set_res2 = hooks.use_state("")
    res_3, set_res3 = hooks.use_state("")

    @event(prevent_default=True)
    def submit_tickers(event):
        print(f"Submitted Tickers {t1}, {t2}, {t3}")
        print(f"Risk Tolerance: {r}")
        print(f"Investment: {i}")
        results = backend.submitTickers(t1, t2, t3, r, i)
        res_raw1 = f"{results[2][0]} : {round(results[0][0] * results[3], 2)} ({results[1][0]}%)"
        res_raw2 = f"{results[2][1]} : {round(results[0][1] * results[3], 2)} ({results[1][1]}%)"
        res_raw3 = f"{results[2][2]} : {round(results[0][2] * results[3], 2)} ({results[1][2]}%)"
        set_res1(res_raw1)
        set_res2(res_raw2)
        set_res3(res_raw3)

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
                    html.div(res_1),
                    html.div(res_2),
                    html.div(res_3)
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
