from models.glance import *
from models.linedp import *
from models.mymodel import *
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_project_releases_dict():
    project_releases_dict = {}
    for release in PROJECT_RELEASE_LIST:
        project = release.split('-')[0]
        if project not in project_releases_dict:
            project_releases_dict[project] = [release]
        else:
            project_releases_dict[project].append(release)

    return project_releases_dict


def run_cross_release_predict(prediction_model):
    for project, releases in get_project_releases_dict().items():
        for i in range(len(releases) - 1):
            print(f'========== {prediction_model.model_name} CR PREDICTION for {releases[i + 1]} ================'[:60])
            model = prediction_model(releases[i], releases[i + 1])

            model.file_level_prediction()
            model.analyze_file_level_result()
            model.line_level_prediction()
            model.analyze_line_level_result()


def run_default():
    # RQ1
    run_cross_release_predict(Glance_LR_Mixed_Sort)
    run_cross_release_predict(Glance_EA_Mixed_Sort)
    run_cross_release_predict(Glance_MD_Mixed_Sort)
    run_cross_release_predict(LineDP_mixedsort)

    run_cross_release_predict(Barinel)
    run_cross_release_predict(Dstar)
    run_cross_release_predict(Ochiai)
    run_cross_release_predict(Op2)
    run_cross_release_predict(Tarantula)

    # RQ2
    run_cross_release_predict(Barinel_without_filevel)
    run_cross_release_predict(Dstar_without_filevel)
    run_cross_release_predict(Ochiai_without_filevel)
    run_cross_release_predict(Op2_without_filevel)
    run_cross_release_predict(Tarantula_without_filevel)

    # RQ3
    run_cross_release_predict(BarinelWithoutCA)
    run_cross_release_predict(DstarWithoutCA)
    run_cross_release_predict(OchiaiWithoutCA)
    run_cross_release_predict(Op2WithoutCA)
    run_cross_release_predict(TarantulaWithoutCA)

    # dis_2
    run_cross_release_predict(BarinelFFLLSort)

    pass


if __name__ == '__main__':
    run_default()
