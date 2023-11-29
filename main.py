import pandas as pd
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------
# a
# sheet

# sinh vien
dfSinhVien = pd.read_excel("./DuLieuThucHanh1.xlsx", sheet_name="Sinh Vien")
# print(dfSinhVien)

# diem so
dfDiemSo = pd.read_excel("./DuLieuThucHanh1.xlsx", sheet_name="Diem So")
# print(dfDiemSo)

# dfDiemSo.info()

# ------------------------------------------------------------------------
# b
df_merged = pd.merge(dfSinhVien, dfDiemSo, on="MSSV")
# print(df_merged)

# ------------------------------------------------------------------------
# c
dfDiemSo.fillna(0, inplace=True)
df_merged.fillna(0, inplace=True)
# print(dfDiemSo)

# ------------------------------------------------------------------------
# d
df_merged["Diem TB"] = df_merged[["Toan", "Ly", "Hoa"]].mean(axis=1)
# print(df_merged)

# ------------------------------------------------------------------------
# e
df_merged["Ket Qua"] = df_merged["Diem TB"].apply(lambda x: "Fail" if x < 5 else "Pass")
# print(df_merged)

# ------------------------------------------------------------------------
# f
max_point = df_merged["Diem TB"].max()
df_SVTopPoint = df_merged[df_merged["Diem TB"] == max_point]
# print(df_SVTopPoint)

# ------------------------------------------------------------------------
# g
df_trung_ten = df_merged.groupby(['Ho Dem', 'Ten']).filter(lambda x: len(x) > 1)
df_trung_ten['Check'] = df_trung_ten['MSSV'].is_unique

# print(df_trung_ten)

# ------------------------------------------------------------------------
# h
# print(df_merged)
plt.bar(df_merged.index, df_merged['Toan'])
plt.xlabel('Index')
plt.ylabel('Toan')
plt.title('Toan cua tat ca SV')
plt.show()

# ------------------------------------------------------------------------
# i
df_merged.to_excel("./DuLieuThucHanhDaTongHop.xlsx", index=False)