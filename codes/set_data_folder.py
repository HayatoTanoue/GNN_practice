import os
import glob
import shutil


# 特定のパラメータネットワークを抽出(判定)する
def check(origin, choose_p):
    origin = origin.split("/")[-1].split("_")

    re = all(
        [
            origin[0] == choose_p["kind"],
            origin[1] in choose_p["node"],
            origin[2] in choose_p["p"],
        ]
    )

    return re


class make_train_data(object):
    def __init__(self, p_s, origin_data_path):
        # 選択するパラメータのリスト
        self.p_s = p_s
        # 画像の保存元
        self.origin_data_path = origin_data_path

    def copy_data(self):
        # pics フォルダの削除
        if os.path.exists("/workspace/my_data/"):
            shutil.rmtree("/workspace/my_data/")

        # 全パスを取得
        paths = glob.glob(self.origin_data_path + "/*.adjlist")
        cnt = 0
        for s in self.p_s:
            save_name = "_".join(
                [",".join(l) if type(l) == list else l for l in list(s.values())]
            )
            save_path = "/workspace/my_data/{}_{}".format(cnt, save_name)
            os.makedirs(save_path)
            cnt += 1
            for path in paths:

                if check(path, s):
                    shutil.copy(path, save_path + "/" + path.split("/")[-1])
