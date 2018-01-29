from collections import OrderedDict
import copy
import functools

import numpy

from skopt import Space as SkoptSpace
from skopt.space import Real
from skopt.utils import check_x_in_space


class Integer(Real):
    def __repr__(self):
        return super(Integer, self).__repr__().replace("Real", "Integer")

    def inverse_transform(self, Xt):
        return super(Integer, self).inverse_transform(Xt).astype(numpy.int)


class Space(object):
    NON_BASE_SPACE = []
    BASE_SPACE = []
    MODELS = {}
    SPACES = {}

    PROFILES = {
        None: {}}

    def __init__(self, model, defaults, opt):
        self.model = model
        self.opt = opt
        self.defaults = defaults
        self.opt.profiles = list(sorted(self.opt.profiles))

    # TODO finish profiles
    def iter_profiles(self):
        return ((profile, self.PROFILES[profile])
                for profile in [None] + self.opt.profiles)

    def validate(self, setting):

        valid = True
        for name, profile in self.iter_profiles():
            for key, value in profile.iteritems():
                valid = valid and setting.get(key) == value

        skopt_space = SkoptSpace(self.get_spaces().values())
        try:
            check_x_in_space(self.dict_to_list(setting), skopt_space)
        except ValueError as e:
            valid = False

        return valid

    def get_spaces(self):
        model_spaces = OrderedDict()

        profiles_hps = sum(
            (profile.keys() for _, profile in self.iter_profiles()),
            [])

        for hp_name, dimension in self.SPACES.iteritems():
            if ((hp_name in self.BASE_SPACE or
                 hp_name in self.MODELS[self.model]) and
                    hp_name not in profiles_hps):

                model_spaces[hp_name] = dimension

        return model_spaces

    def force_profiles(self, setting):
        for name, profile in self.iter_profiles():
            for key, value in profile.iteritems():
                setting[key] = value

    def force_options(self, setting):
        # setting["verbose"] = self.opt.verbose_process
        # setting["print_progress"] = self.opt.debug
        setting["gpu_id"] = self.opt.gpu_id

    def list_to_dict(self, space):
        setting = dict(zip(self.get_spaces().keys(), space))
        self.force_options(setting)
        self.force_profiles(setting)

        return setting

    def dict_to_list(self, setting):
        setting = setting.copy()

        space = [setting[hp_name] for hp_name in self.get_spaces().iterkeys()]
        return space

    def get_default(self):
        setting = copy.copy(self.defaults)
        self.force_options(setting)
        self.force_profiles(setting)

        return setting

    def get_validate_sample_fct(self):
        return functools.partial(self._validate_sample, self)

    @staticmethod
    def _validate_sample(space, row):
        setting = space.list_to_dict(row)

        skopt_space = SkoptSpace(space.get_spaces().values())
        try:
            check_x_in_space(row, skopt_space)
        except ValueError:
            return False

        has_activations_covariance_penalty = (
            setting.get("activations_covariance_penalty", 0.) >
            COEFFICIENT_LIMIT)
        if (has_activations_covariance_penalty and
                (setting.get("normalized_activations", "NONE") != "NONE")):
            return False

        if (setting.get("pre_projection_train_simultaneously") and
                setting.get("pre_projection_train_alternatively")):
            return False

        # Penalty type must be chosen
        if (setting.get("pre_projection_train_alternatively") and
                setting.get("pre_projection_training_penalty") is None):
            return False

        for level in ["projections", "activations"]:
            rescaled_key = "rescaled_%s" % level
            normalized_key = "normalized_%s" % level
            centered_key = "centered_%s" % level
            # epsilon_key = "normalized_epsilon_%s" % level

            if setting.get(rescaled_key) == "ALL":
                if setting.get(normalized_key) != "ALL":
                    return False

            if setting.get(rescaled_key) == "FEATURES":
                if (setting.get(normalized_key)
                        not in ["FEATURES", "ALL"]):
                    return False

            if setting.get(normalized_key) == "ALL":
                if setting.get(centered_key) != "ALL":
                    return False

            if setting.get(normalized_key) == "FEATURES":
                # Library's version always centers and normalizes
                if setting.get("force_library_batch_norm"):
                    if setting.get(centered_key) != "FEATURES":
                        return False
                else:
                    if (setting.get(centered_key)
                            not in ["FEATURES", "ALL"]):
                        return False

            if setting.get(normalized_key) == "NONE":
                if setting.get("force_library_batch_norm"):
                    if setting.get(centered_key) != "NONE":
                        return False

        return True
