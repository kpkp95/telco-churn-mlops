from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from src.serving.inference import predict


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in the telecom industry",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "ok"}


class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str

    PhoneService: str
    MultipleLines: str

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        return predict(data.model_dump())
    except Exception as e:
        return {"error": str(e)}


def gradio_interface(
    gender,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges,
):
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    result = predict(data)

    return (
        f"Prediction: {result['prediction']}\n"
        f"Churn probability: {result['churn_probability']}\n"
        f"Threshold: {result['threshold']}"
    )


with gr.Blocks(
    title="Telco Customer Churn Predictor",
    css="""
    .gradio-container {
        max-width: 1150px !important;
        margin: 0 auto !important;
        padding-top: 10px !important;
    }

    h1 {
        font-size: 30px !important;
        margin-bottom: 6px !important;
        text-align: center;
    }

    .small-desc {
        font-size: 14px;
        color: #cfcfcf;
        max-width: 900px;
        margin: 0 auto 18px auto;
        text-align: center;
        line-height: 1.5;
    }
    """
) as demo:
    gr.Markdown(
        """
        <h1>Telco Customer Churn Predictor</h1>
        <div class="small-desc">
            Enter a customer’s telecom service and billing details to estimate churn risk.
            The model returns a churn probability and flags the customer as likely to churn
            when the predicted probability is greater than or equal to the decision threshold.
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
            phone_service = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
            multiple_lines = gr.Dropdown(
                ["Yes", "No", "No phone service"],
                label="Multiple Lines",
                value="No",
            )
            internet_service = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service",
                value="Fiber optic",
            )
            online_security = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Security",
                value="No",
            )
            online_backup = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Backup",
                value="No",
            )
            device_protection = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Device Protection",
                value="No",
            )

        with gr.Column():
            tech_support = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Tech Support",
                value="No",
            )
            streaming_tv = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming TV",
                value="Yes",
            )
            streaming_movies = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming Movies",
                value="Yes",
            )
            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract",
                value="Month-to-month",
            )
            paperless_billing = gr.Dropdown(
                ["Yes", "No"],
                label="Paperless Billing",
                value="Yes",
            )
            payment_method = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                label="Payment Method",
                value="Electronic check",
            )
            tenure = gr.Number(label="Tenure Months", value=1, minimum=0, maximum=100)
            monthly_charges = gr.Number(label="Monthly Charges", value=85.0, minimum=0, maximum=200)
            total_charges = gr.Number(label="Total Charges", value=85.0, minimum=0, maximum=10000)

    with gr.Row():
        clear_btn = gr.Button("Clear", scale=1)
        submit_btn = gr.Button("Submit", variant="primary", scale=1)

    output_box = gr.Textbox(label="Churn Prediction", lines=3)

    gr.Examples(
        examples=[
            [
                "Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
                "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
                1, 85.0, 85.0,
            ],
            [
                "Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
                "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
                60, 45.0, 2700.0,
            ],
        ],
        inputs=[
            gender,
            partner,
            dependents,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            tenure,
            monthly_charges,
            total_charges,
        ],
    )

    submit_btn.click(
        fn=gradio_interface,
        inputs=[
            gender,
            partner,
            dependents,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            tenure,
            monthly_charges,
            total_charges,
        ],
        outputs=output_box,
    )

    clear_btn.click(
        fn=lambda: [
            "Male",
            "No",
            "No",
            "Yes",
            "No",
            "Fiber optic",
            "No",
            "No",
            "No",
            "No",
            "Yes",
            "Yes",
            "Month-to-month",
            "Yes",
            "Electronic check",
            1,
            85.0,
            85.0,
            "",
        ],
        inputs=[],
        outputs=[
            gender,
            partner,
            dependents,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            tenure,
            monthly_charges,
            total_charges,
            output_box,
        ],
    )


app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui",
)