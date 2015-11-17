import pandas as pd
import datetime

def save_result(index, result):
    df = pd.DataFrame({"PostId": index, "OpenStatus": result})
    df.to_csv("output/" + str(datetime.datetime.now()), index=False)

preds1 = pd.read_csv("output/preds3")["OpenStatus"].values
preds2 = pd.read_csv("output/preds4")["OpenStatus"].values
post_id = pd.read_csv("output/preds3")["PostId"].values

preds = 0.7*preds1 + 0.3*preds2

save_result(post_id, preds)
