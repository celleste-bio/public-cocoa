"""
exploring the effects of breeding
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)

    ############################################################
    
    # fat data
    fat_info = pd.read_sql_query(
        """
        SELECT *
        FROM butterfat
        JOIN ref_info ON butterfat.Refcode = ref_info.refcode;
        """,
        connection
    )
    fat_info = fat_info[["Clone name", "Refcode", "Fat", "year"]].dropna()
    fat_info["Fat"] = pd.to_numeric(fat_info["Fat"], errors='coerce')
    fat_info = fat_info.dropna()

    grouped_data = fat_info.groupby('year')['Fat'].agg(['mean', 'std'])

    plt.plot(grouped_data.index, grouped_data['mean'], label='Mean Line', color='red')

    plt.fill_between(
        grouped_data.index,
        grouped_data['mean'] - grouped_data['std'],
        grouped_data['mean'] + grouped_data['std'],
        facecolor='blue', alpha=0.3, label='Std Dev Ribbon'
    )

    plt.xlabel('Year')
    plt.ylabel('Fat Precentage')
    plt.title('Mean Fat Precentage Over Years with Standard Deviation')
    plt.legend()
    plt.grid(False)
    plt.show()

    ############################################################
    
    query = """
        SELECT "yield"."Clone name", "yield"."Refcode", "yield"."Yield ha", "butterfat"."Fat", "yield"."Pod index"
        FROM "yield"
        JOIN "butterfat" ON "yield"."Refcode" = "butterfat"."Refcode" AND "yield"."Clone name" = "butterfat"."Clone name";
    """

    yield_fat_info = pd.read_sql_query(query, connection)

    columns_to_process = ["Yield ha", "Fat", "Pod index"]

    for column in columns_to_process:
        yield_fat_info[column] = pd.to_numeric(yield_fat_info[column], errors='coerce')
        yield_fat_info.dropna(subset=[column], inplace=True)
        yield_fat_info.sort_values(by=[column], ascending=False, inplace=True)
        yield_fat_info.reset_index(drop=True, inplace=True)

    plt.scatter(yield_fat_info["Yield ha"], yield_fat_info["Fat"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(yield_fat_info["Yield ha"], yield_fat_info["Fat"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(yield_fat_info["Yield ha"])
    r_squared = r2_score(yield_fat_info["Fat"], y_pred)

    # Plot the regression line
    plt.plot(yield_fat_info["Yield ha"], poly_line(yield_fat_info["Yield ha"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Yield ha")
    plt.ylabel("Fat")
    plt.title("Scatter Plot of Yield ha vs Fat")

    # Show the plot
    plt.legend()
    plt.show()


    plt.scatter(yield_fat_info["Pod index"], yield_fat_info["Fat"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(yield_fat_info["Pod index"], yield_fat_info["Fat"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(yield_fat_info["Pod index"])
    r_squared = r2_score(yield_fat_info["Fat"], y_pred)

    # Plot the regression line
    plt.plot(yield_fat_info["Pod index"], poly_line(yield_fat_info["Pod index"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Pod index")
    plt.ylabel("Fat")
    plt.title("Scatter Plot of Pod index vs Fat")

    # Show the plot
    plt.legend()
    plt.show()

    ############################################################

    query = """
        SELECT "Clone name", "Refcode", "Yield ha", "Yield tree", "Pod index", "Seed index"
        FROM yield;
    """

    yield_info = pd.read_sql_query(query, connection)

    # columns_to_process = ["Yield ha", "Yield tree", "Pod index", "Seed index"]
    columns_to_process = ["Yield ha", "Yield tree"]

    for column in columns_to_process:
        yield_info[column] = pd.to_numeric(yield_info[column], errors='coerce')
        yield_info.dropna(subset=[column], inplace=True)
        yield_info.sort_values(by=[column], ascending=False, inplace=True)
        yield_info.reset_index(drop=True, inplace=True)

    plt.plot(yield_info["Yield ha"])
    plt.xlabel("Index")
    plt.ylabel("Yield ha")
    plt.title("Line Plot of Yield ha")
    plt.show()

    yield_info.iloc[8:43, :]

    plt.plot(yield_info["Yield tree"])
    plt.xlabel("Index")
    plt.ylabel("Yield tree")
    plt.title("Line Plot of Yield tree")
    plt.show()



    plt.scatter(yield_info["Yield ha"], yield_info["Pod index"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(yield_info["Yield ha"], yield_info["Pod index"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(yield_info["Yield ha"])
    r_squared = r2_score(yield_info["Pod index"], y_pred)

    # Plot the regression line
    plt.plot(yield_info["Yield ha"], poly_line(yield_info["Yield ha"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Yield ha")
    plt.ylabel("Pod index")
    plt.title("Scatter Plot of Yield ha vs Pod index")

    # Show the plot
    plt.legend()
    plt.show()

    
    plt.scatter(yield_info["Seed index"], yield_info["Pod index"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(yield_info["Seed index"], yield_info["Pod index"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(yield_info["Seed index"])
    r_squared = r2_score(yield_info["Pod index"], y_pred)

    "Pod index", "Seed index"
    # Plot the regression line
    plt.plot(yield_info["Seed index"], poly_line(yield_info["Seed index"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Seed index")
    plt.ylabel("Pod index")
    plt.title("Scatter Plot of Seed index vs Pod index")

    # Show the plot
    plt.legend()
    plt.show()


    plt.scatter(yield_info["Yield ha"], yield_info["Yield tree"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(yield_info["Yield ha"], yield_info["Yield tree"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(yield_info["Yield ha"])
    r_squared = r2_score(yield_info["Yield tree"], y_pred)

    "Yield tree", "Yield ha"
    # Plot the regression line
    plt.plot(yield_info["Yield ha"], poly_line(yield_info["Yield ha"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Yield ha")
    plt.ylabel("Yield tree")
    plt.title("Scatter Plot of Yield ha vs Yield tree")

    # Show the plot
    plt.legend()
    plt.show()






    # yield data
    yield_info = pd.read_sql_query(
        """
        SELECT *
        FROM yield
        JOIN ref_info ON yield.Refcode = ref_info.refcode;
        """,
        connection
    )
    yield_info = yield_info[["Clone name", "Refcode", "Yield ha", "year"]].dropna()
    yield_info["Yield ha"] = pd.to_numeric(yield_info["Yield ha"], errors='coerce')
    yield_info = yield_info.dropna()

    grouped_data = yield_info.groupby('year')['Yield ha'].agg(['mean', 'std'])

    plt.plot(grouped_data.index, grouped_data['mean'], label='Mean Line', color='red')

    plt.fill_between(
        grouped_data.index,
        grouped_data['mean'] - grouped_data['std'],
        grouped_data['mean'] + grouped_data['std'],
        facecolor='blue', alpha=0.3, label='Std Dev Ribbon'
    )

    plt.xlabel('Year')
    plt.ylabel('Yield ha')
    plt.title('Mean Yield ha Over Years with Standard Deviation')
    plt.legend()
    plt.grid(False)
    plt.show()

    ############################################################

    query = """
        SELECT "bean"."Clone name", "bean"."Refcode", "bean"."Cotyledon dry weight", "bean"."Total dry weight", "butterfat"."Fat"
        FROM "bean"
        JOIN "butterfat" ON "bean"."Refcode" = "butterfat"."Refcode" AND "bean"."Clone name" = "butterfat"."Clone name";
    """

    bean_fat_info = pd.read_sql_query(query, connection)

    columns_to_process = ["Total dry weight", "Fat"]

    for column in columns_to_process:
        bean_fat_info[column] = pd.to_numeric(bean_fat_info[column], errors='coerce')
        bean_fat_info.dropna(subset=[column], inplace=True)
        bean_fat_info.sort_values(by=[column], ascending=False, inplace=True)
        bean_fat_info.reset_index(drop=True, inplace=True)

    plt.scatter(bean_fat_info["Total dry weight"], bean_fat_info["Fat"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(bean_fat_info["Total dry weight"], bean_fat_info["Fat"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(bean_fat_info["Total dry weight"])
    r_squared = r2_score(bean_fat_info["Fat"], y_pred)

    # Plot the regression line
    plt.plot(bean_fat_info["Total dry weight"], poly_line(bean_fat_info["Total dry weight"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Total dry weight")
    plt.ylabel("Fat")
    plt.title("Scatter Plot of Total dry weight vs Fat")

    # Show the plot
    plt.legend()
    plt.show()

############################################################

    query = """
        SELECT "bean"."Clone name", "bean"."Refcode", "bean"."Total dry weight", "yield"."Yield ha"
        FROM "bean"
        JOIN "yield" ON "bean"."Refcode" = "yield"."Refcode" AND "bean"."Clone name" = "yield"."Clone name";
    """

    bean_yield_info = pd.read_sql_query(query, connection)

    columns_to_process = ["Total dry weight", "Yield ha"]

    for column in columns_to_process:
        bean_yield_info[column] = pd.to_numeric(bean_yield_info[column], errors='coerce')
        bean_yield_info.dropna(subset=[column], inplace=True)
        bean_yield_info.sort_values(by=[column], ascending=False, inplace=True)
        bean_yield_info.reset_index(drop=True, inplace=True)

    plt.scatter(bean_yield_info["Total dry weight"], bean_yield_info["Fat"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(bean_yield_info["Total dry weight"], bean_yield_info["Fat"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(bean_yield_info["Total dry weight"])
    r_squared = r2_score(bean_yield_info["Fat"], y_pred)

    # Plot the regression line
    plt.plot(bean_yield_info["Total dry weight"], poly_line(bean_yield_info["Total dry weight"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Total dry weight")
    plt.ylabel("Fat")
    plt.title("Scatter Plot of Total dry weight vs Fat")

    # Show the plot
    plt.legend()
    plt.show()

############################################################

    query = """
        SELECT "bean"."Clone name", "bean"."Refcode", "bean"."Total dry weight", "butterfat"."fat"
        FROM "bean"
        JOIN "butterfat" ON "bean"."Refcode" = "butterfat"."Refcode" AND "bean"."Clone name" = "butterfat"."Clone name";
    """

    bean_fat_info = pd.read_sql_query(query, connection)

    columns_to_process = ["Total dry weight", "Fat"]

    for column in columns_to_process:
        bean_fat_info[column] = pd.to_numeric(bean_fat_info[column], errors='coerce')
        bean_fat_info.dropna(subset=[column], inplace=True)
        bean_fat_info.sort_values(by=[column], ascending=False, inplace=True)
        bean_fat_info.reset_index(drop=True, inplace=True)

    plt.scatter(bean_fat_info["Total dry weight"], bean_fat_info["Fat"], label="Data Points")

    # Fit a linear regression line
    coefficients = np.polyfit(bean_fat_info["Total dry weight"], bean_fat_info["Fat"], 1)
    poly_line = np.poly1d(coefficients)
    y_pred = poly_line(bean_fat_info["Total dry weight"])
    r_squared = r2_score(bean_fat_info["Fat"], y_pred)

    # Plot the regression line
    plt.plot(bean_fat_info["Total dry weight"], poly_line(bean_fat_info["Total dry weight"]), color='red', label=f"Regression Line: {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

    # Labeling the axes and title
    plt.xlabel("Total dry weight")
    plt.ylabel("Fat")
    plt.title("Scatter Plot of Total dry weight vs Fat")

    # Show the plot
    plt.legend()
    plt.show()


    ############################################################

    connection.close()

if __name__ == "__main__":
    main()

