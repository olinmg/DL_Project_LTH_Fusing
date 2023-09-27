import json
import os
import clipboard

def get_res(res):
    IF = "{:.2f}".format(res["IF"])
    default = "{:.2f}".format(res["default"])
    diff = res["IF"] - res["default"]
    print("diff is: ", diff)
    if diff >= 0:
        diff = "+{:.2f}".format(diff)
    else:
        diff = "{:.2f}".format(diff)

    if abs(res["IF"] - res["default"]) <= 0.5:
        return f" & {default} & {IF} & {diff}"
    elif res["IF"] - res["default"] > 0:
        return f" & {default} & " + """\\textbf{""" + IF + """}""" + f" & \\textcolor" + """{darkgreen}{""" + diff + """}"""
    else:
        return " & " + """\\textbf{""" + default + """}""" + f" & {IF} & \\textcolor" + """{lightred}{""" + diff + """}"""
    if (res["IF"] - res["default"]) <= 0.01:
        return f" & {default} & {IF} & {diff}"
    elif IF > default:
        return f" & {default} & " + """\\textbf{""" + IF + """}""" + f" & {diff}"
    else:
        return " & " + """\\textbf{""" + default + """}""" + f" & {IF} & {diff}"


start_table_str = """\\begin{table}\n\\begin{scriptsize}\n\caption{Intra-Fusion vs. default pruning with finetuning.}\n\label{sample-table}\n\centering\n"""
start_table_str += """\\begin{tabular}{lllllllllll}\n\\toprule\n"""
start_table_str += """Group & Sparsity (\%) & $\ell_1$(\%) & IF ($\ell_1$)(\%) & $\delta$(\%) & Taylor(\%) & IF (Taylor)(\%) & $\delta$(\%) & LAMP(\%) & IF (LAMP)(\%) & $\delta$(\%) \\\\\n"""
start_table_str += """\midrule\n"""

end_table_str = "\\bottomrule\n" + """\end{tabular}\n\end{scriptsize}\n\end{table}"""


path_to_json = './DataFree/DataFree/IMAGENET/Resnet50/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)

l1_jsons = []
taylor_jsons = []
lamp_jsons = []

for json_file in json_files:
    if "l1" in json_file:
        l1_jsons.append(json_file)
    if "taylor" in json_file:
        taylor_jsons.append(json_file)
    if "lamp" in json_file:
        lamp_jsons.append(json_file)

with open(path_to_json+l1_jsons[0], 'r') as f:
    l1_res = json.load(f)["l1"]
with open(path_to_json+taylor_jsons[0], 'r') as f:
    taylor_res = json.load(f)["taylor"]
with open(path_to_json+lamp_jsons[0], 'r') as f:
    lamp_res = json.load(f)["lamp"]

sparsities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
table_str = start_table_str
for i, group_idx in enumerate(l1_res.keys()):
    if i > 10:
        break
    if i == 6:
        table_str += end_table_str
        table_str += start_table_str

    group_str = """\multirow{7}{*}{Group """ + group_idx + """}"""
    print(group_str)
    for sparsity in sparsities:
        group_str += " & {sp}".format(sp = int(sparsity*100))

        group_str += get_res(l1_res[group_idx][str(sparsity)])
        group_str += get_res(taylor_res[group_idx][str(sparsity)])
        group_str += get_res(lamp_res[group_idx][str(sparsity)]) + "\\\\[1mm]"
    
    if i < len(list(l1_res.keys()))-1 and i != 5 and i != 10:
        group_str += "\n\midrule\n"

    table_str += group_str


table_str += "\\bottomrule\n" + """\end{tabular}\n\end{scriptsize}\n\end{table}"""

print(table_str)
clipboard.copy(table_str)









