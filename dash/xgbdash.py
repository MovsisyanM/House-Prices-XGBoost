import os
import sys


sys.path.insert(0, os.path.dirname(__file__))
 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output 
import plotly.express as px 
import pandas as pd
import numpy as np
from pathlib import Path
 
app = dash.Dash(serve_locally=False, requests_pathname_prefix="/hppxgb/", routes_pathname_prefix = "/") 
path = Path().resolve()/"data" 
trainx = pd.read_csv(path/"train.csv", index_col="Id") 

numeric_cols_desc = [
    "Linear feet of street connected to property",
    "Lot size in square feet",
    "Original construction date",
    "Remodel date",
    "Masonry veneer area in square feet",
    "Type 1 finished square feet",
    "Type 2 finished square feet",
    "Unfinished square feet of basement area",
    "Total square feet of basement area",
    "First Floor square feet",
    "Second floor square feet",
    "Low quality finished square feet (all floors)",
    "Above grade (ground) living area square feet",
    "Year garage was built",
    "Size of garage in square feet",
    "Wood deck area in square feet",
    "Open porch area in square feet",
    "Enclosed porch area in square feet",
    "Three season porch area in square feet",
    "Screen porch area in square feet",
    "Pool area in square feet",
    "$ Value of miscellaneous feature",
    "Year Sold"
]

cat_cols_desc = [
    "Identifies the type of dwelling involved in the sale.",
    "Identifies the general zoning classification of the sale.",
    "Type of road access to property",
    "Type of alley access to property",
    "General shape of property",
    "Flatness of the property",
    "Type of utilities available",
    "Lot configuration",
    "Slope of property",
    "Physical locations within Ames city limits",
    "Proximity to various conditions",
    "Proximity to various conditions (if more than one is present)",
    "Type of dwelling",
    "Style of dwelling",
    "Rates the overall material and finish of the house",
    "Rates the overall condition of the house",
    "Type of roof",
    "Roof material",
    "Exterior covering on house",
    "Second exterior covering on house",
    "Masonry veneer type",
    "Evaluates the quality of the material on the exterior",
    "Evaluates the present condition of the material on the exterior",
    "Type of foundation",
    "Evaluates the height of the basement",
    "Evaluates the general condition of the basement",
    "Refers to walkout or garden level walls",
    "Rating of basement finished area",
    "Rating of basement finished area (if multiple types)",
    "Type of heating",
    "Heating quality and condition",
    "Central air conditioning",
    "Electrical system",
    "Basement full bathrooms",
    "Basement half bathrooms",
    "Full bathrooms above grade",
    "Half baths above grade",
    "Bedrooms above grade",
    "Kitchens above grade",
    "Kitchen quality",
    "Total rooms above grade",
    "Home functionality",
    "Number of fireplaces",
    "Fireplace quality",
    "Garage location",
    "Interior finish of the garage",
    "Size of garage in car capacity",
    "Garage quality",
    "Garage condition",
    "Paved driveway",
    "Pool quality",
    "Fence quality",
    "Miscellaneous feature not covered in other categories",
    "Month Sold (MM)",
    "Type of sale",
    "Condition of sale"

]

numeric_cols = [col for col in trainx if trainx[col].dtype ==
                "int64" or trainx[col].dtype == "float64"][1:]
numeric_cols.remove("OverallQual")
numeric_cols.remove("OverallCond")
numeric_cols.remove("BsmtFullBath")
numeric_cols.remove("BsmtHalfBath")
numeric_cols.remove("FullBath")
numeric_cols.remove("HalfBath")
numeric_cols.remove("BedroomAbvGr")
numeric_cols.remove("KitchenAbvGr")
numeric_cols.remove("MoSold")
numeric_cols.remove("GarageCars")
numeric_cols.remove("Fireplaces")
numeric_cols.remove("TotRmsAbvGrd")

# Generating selection options for dynamic plots
opts = [{"label": numeric_cols_desc[i], "value": numeric_cols[i]}
        for i in range(len(numeric_cols) - 1)]
cat_cols = [col for col in trainx.columns if col not in numeric_cols]
opts2 = [{"label": cat_cols_desc[i], "value": cat_cols[i]}
         for i in range(len(cat_cols) - 1)]
opts.extend(opts2)

# Create the server handle
server = app.server
application = server

# Creating the html layout of the app
app.layout = html.Div([
    html.Div(),
    html.Div([
        html.Div([
            html.Article([
                html.H1("XGBoost - House Price Prediction"),
                html.H6("2021 May"),
                html.P([
                    """\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Ask a home buyer to describe their dream house, and they probably won't 
                    begin with the height of the basement ceiling or the proximity to an 
                    east-west railroad. But this playground competition's dataset proves 
                    that much more influences price negotiations than the number of bedrooms 
                    or a white-picket fence.  With """,
                    html.B(["79 explanatory variables"]),
                    """describing (almost) every aspect of residential homes in """,
                    html.B(["Ames, Iowa"]),
                    """, this competition challenged us to predict the final price of each home.""",
                    html.Br(),
                    html.Br(),
                    """\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0Esfira Babajanyan, Hrach Yeghiazaryan and I accepted this challenge. 
                    Although this was our first-ever model, we charged into the unknown 
                    head-on. We built a pipeline capable of determining which features 
                    are useless, capable of generating composite features from existing 
                    ones. The architecture of the pipeline proved to be flexible, allowing 
                    us to do various operations with a single call, for example, grid-searching. 
                    The charge into the unknown made us learn a vast amount from the new 
                    challenges we were facing. I hope everybody gets to experience the 
                    joy of challenging oneself.""",
                    html.Br(), html.Br(), html.Br(),
                ])
            ])
        ]),
        html.Hr(),
        html.Div([
            html.Br(),
            html.Br(),
            html.H3(["Feature versus Target Variable"]),
            dcc.Dropdown(
                options=opts,
                multi=True,
                value=["1stFlrSF"],
                id="input"
            ),
            html.Div(id="output")
        ]),
        html.Hr(),
        html.Div([
            html.Br(),
            html.Br(),
            html.H3("The Structure of the Pipeline"),
            html.P([
                "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0The pipeline consists of four steps: the Column Remover, the Feature Generator, the One-Hot Encoder and the Extreme Gradient Boost regressor. ",
                html.Br(),
                "The first step, ",
                html.B("Column Remover"),
                ", can be given a list of column names to remove from the data, not letting it pass to the next stage of the pipeline. This pipeline component is beneficial since it allows us to run cross-validation checks on the entire pipeline and compare the scores to other cross-validations run on the pipeline with one column removed, thereby quickly determining whether the removed column affected the score positively or negatively.",
                html.Br(),
                "The second step, ",
                html.B("Feature Generator"),
                ", generates new features from the existing ones. It takes a list of sub-generator functions, which take a data frame and output a dataframe with the additional composite feature. The Feature Generator passes the data to each of the sub-generators and outputs the resulting dataframe. If enough ingredient features are missing from the dataframe due to the Column Remover, the new composite feature is not created.",
                html.Br(),
                "The third step, ",
                html.B("One-Hot Encoder"),
                ", takes the columns that are not numeric and one-hot encodes them. This step is necessary since the final estimator only accepts numerical data.",
                html.Br(),
                "The final step, ",
                html.B("Extreme Gradient Boost regressor"),
                ", learns from the training data provided to it and generates trees that can collectively predict house prices."
            ]),
            html.Img(src=r"https://movsisyan.info/resources/flowchart.svg")
        ]),
        html.Hr(),
        html.Div([
            html.Br(),
            html.Br(),
            html.H3("The Team"),
            html.P([
                "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0We are a team of students eager to jump into the data science world. With this project, we hoped to learn as much as we could about the challenges data scientists are faced with every day. Luckily, the project was a great success, and it met all of our expectations. We learned what pipelines are and their usefulness, we learned how XGBoost works in theory, we got to tune hyperparameters. We faced many problems which we didn't know we were going to face, helping us further develop our problem-solving skills. This project will stand to be the portal to data science for our team.",
                html.Br(), html.Br(),
                "Esfira Babajanyan: Data Visualizations, Presentation",
                html.Br(),
                "Hrach Yeghyazaryan: Data Visualizations, Presentation",
                html.Br(),
                "Mher Movsisyan: Building the pipeline, Dash-based Project Website",
                html.Br(), html.Br(),
                html.A("Click here to visit the project GitHub repository", target="_blank",
                       href="https://github.com/MovsisyanM/House-Prices-XGBoost"),
                html.Br(),
                html.A("Click here to visit the kaggle notebook", target="_blank",
                       href="https://www.kaggle.com/movsisyanm/house-prices-xgboost")
            ])
        ]),
    ])
], className="section")

# Called every time the selection is updated


@app.callback(
    Output(component_id="output", component_property="children"),
    [Input(component_id="input", component_property="value")]
)
def update_value(input_data):
    """Takes a list of column names, returns a list of Graphs"""
    graphs = []
    for col in input_data:
        if col in cat_cols:
            fig = px.violin(trainx, y="SalePrice", x=col, box=True,
                            points="all", hover_data=trainx.columns)
        else:
            fig = px.scatter(trainx, x=col, y="SalePrice", trendline="ols")
        graphs.append(dcc.Graph(figure=fig))
    return graphs


if __name__ == '__main__':
    # Starting the server
    app.run_server(debug=False)

