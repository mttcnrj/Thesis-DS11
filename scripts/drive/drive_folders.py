"""Google Drive folder ids of the `Thesis II` data folder (shared link given by the team).

Used only by scripts/drive/*.py to rebuild `data/raw/` with the same relative paths as on Drive
when the Drive desktop client is not available (e.g. a cloud container). On the chosen laptop,
`scripts/sync_drive.py` copies from the mounted Drive folder instead.
"""
ROOT = "1BY52VcZfSQAx4Il4xbqTKD4BQGr9Wws_"

# folder id -> path relative to the Drive root folder
FOLDERS = {
    "1eR64_Gxdv097_a1-1wT2x5kHDqibTY2j": "MMDA Dataset",
    "1nopXHCx75M97XzeD_zOH-cJiJcaIFo3_": "PAGASA Dataset",
    "1jK-hRP_QQgS9MM77m1LYeB5W7uZJR9Yj": "Kaggle Dataset",
    "1P9aEwlAu4ZtJpIO4Lx3QLkCG9qg9ZvXS": "MMDA Dataset/WOCR and Water elevation reading of Aviles and Quiapo Pumping Station",
    "1iD52VBzO0XAM0r66vxrxnzRPOV-O5kr3": "MMDA Dataset/Waterlevel Data 2018-2024",
}
_WL = "MMDA Dataset/Waterlevel Data 2018-2024"
_STATIONS = {
    "Pandacan 2018-2024": {
        "1HMYv_eEeNptfJuFN6FEqxQI0Rhw5YV9R": "2018", "1dhOUR7RBKQ-MZvOUdFSC1jvN4q_SUx7r": "2019",
        "1IKL0v04rqAkz_E6VCVt1F1ehJmtL8_4J": "2020", "1QyK9F_JE50dg63Zkv8huq4qCXGl46Zy6": "2021",
        "11CPjD2cXsv8HAXlKq3gwSfunKetlw0w_": "2022", "1AtSj0qVRvSKM3yvV2Y_vduWZIhmAw9Yw": "2023",
        "1XKSl0Fj4gD4iqMMoDzMBXF5pAm1g20Yw": "2024"},
    "Sto. Niño 2018-2024": {
        "1MT_bMwisk35bKWQils-Qms2x5F8q3968": "2018", "1ZEhCdPYsaLr2sryMSorvRE7-x016ahww": "2019",
        "1Lg6QwYdjwiaxLF___ByhLWzJ8qASw94F": "2020", "19SXCigTc-zskgrAPDMuK76JBLIWvtCm_": "2021",
        "1Lr3n1c7cc7Vye2xwGV0qRhDtDWaYA-6f": "2022", "10vH8ueK4WqglVZFPRigRo92SJeUhrEMn": "2023",
        "1ehxe_CxVyJ4qxg3j4Z2J6r1KRDfGQ4Bf": "2024"},
    "Montalban 2018-2024": {
        "1C2FK_9YrOrzPigm8gy1MkFLtdb81v0Rp": "2018", "1KC-FEnOtu1RIBdyUA4wAkEJSBpH34E1P": "2019",
        "1HqPge2M5D8BWiEofw2P_mdx3ZESJBgG1": "2020", "1iPjXpTt4-1X3bq5OOzZC17lU-a99fS5N": "2021",
        "1KE21zOV_EbuV46keKoC4CUSweKKPerag": "2022", "1WXc44m8CV8ijlCgLRGYOMq3tMnBNRIQ0": "2023",
        "1JAQJbW4dY0QymZpKxqhKqVB-_LAICj2X": "2024"},
    "Rosario JS 2018-2024": {
        "1kT-YBhZzKMaE7bwCkP7DmCYQjoUEGMKw": "2018", "1T8aLACXZrrmfrQan033IJpsbF1GR-Cg3": "2019",
        "19GpGHuJNZJ7_-E05P40vyRRR0MjuaVgz": "2020", "1SKQ4a5SiXzuhpB5RxVapVmSKt86nk7wf": "2021",
        "1WxmGZ2BDi25PawCQccWLG6VERu6dRo5q": "2022", "1NW1xu2_qtg_luX35yjT12_xL9mzEk5UX": "2023",
        "1vijoS8iU-XreHJMtALLpMvLuQQXdESnw": "2024"},
    "Fort Santiago 2018-2024": {
        "1Tyi67716toQ9LOuGOr2StBBNyMXngSQm": "2018", "1pyA4kq9TxPfRfTwTUO9fqC9bdMTZBaxj": "2019",
        "1PhT--IFnZbXw3DOxG_KH-n4aODNBgmLy": "2020", "1WR4ZwLYNZCXy3t9Mcbw5Qa-9usxeVjVC": "2021",
        "15xuQIhDwlr5ALocbmHr60L6eR79P2DsM": "2022", "1vTQeEB3crFztItVGK5DalX5EFnoKA1x4": "2023",
        "1_7gRDYmJq0uTHkOl2NaD-vOwPXNhhI6q": "2024"},
    "Rosario LS 2018-2020": {
        "1t7EgRb9QdZKoS0s-aA9Pjyb1ZGrUPDxq": "Rosario Lake Side 2018",
        "1YAfSj_F8MGxohH3ZoNXmz5oPYRIkXCDJ": "Rosario Lake Side 2019",
        "1BNa0Q0RTPGWDk_HLa_N7QuvxWJxIndC7": "Rosario Lake Side 2020"},
}
STATION_FOLDERS = {
    "1di8Pki1EImMP-T4QKTON88QlqWm4Az59": "Pandacan 2018-2024",
    "1duXokHuOky4DhmjPImYiwKR13N2qTfNR": "Sto. Niño 2018-2024",
    "1RN5mdvdPTitdwR3R5af_BlGDi2gutoSQ": "Montalban 2018-2024",
    "1U-UlVqdaqwPDkWXc8Hc80p1RkUwv7nuR": "Rosario JS 2018-2024",
    "1gerqZLDDCW6nnvR5Ss3F8EVdR8p7zW11": "Fort Santiago 2018-2024",
    "1G-G92Qqp5s5hTkZxv1_pdQ_0Y48klVIU": "Rosario LS 2018-2020",
}
for _sid, _st in STATION_FOLDERS.items():
    FOLDERS[_sid] = f"{_WL}/{_st}"
    for _fid, _yr in _STATIONS[_st].items():
        FOLDERS[_fid] = f"{_WL}/{_st}/{_yr}"

# Rosario JS keeps its monthly files one level deeper (year/<subfolder>/).
_RJS = f"{_WL}/Rosario JS 2018-2024"
FOLDERS.update({
    "1eu8n_EhPJv6bgqw-_m8wuhJ7QRL8BnkG": f"{_RJS}/2018/ROSARIO JS 2018",
    "118eFoiBZdwf9w4h5h8QR0AfLXj8tdtlR": f"{_RJS}/2019/Rosario JS 2019",
    "1CZ_3WYd2_TA5jfSXSuFGEQnzwgZX2RzB": f"{_RJS}/2020/Rosario JS 2020",
    "1-eSO-Mubm5GuVglZWAMwakFsaUdpUA4I": f"{_RJS}/2021/Rosario JS 2021 VERIFIED",
    "10SmVFCMQL7I2zU-EiqlXjCaI7ZX7BrBI": f"{_RJS}/2022/Rosario JS 2022",
    "1dZOp7-__D8Gqs52ERZBcHZpMjQvehh7S": f"{_RJS}/2023/Rosario JS",
    "1SFwUz-5rgAoy3SSdmNuProb7GwJ_0EMM": f"{_RJS}/2024/Rosario JS 2024",
})
