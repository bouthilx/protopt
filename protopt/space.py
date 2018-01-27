from collections import OrderedDict
import functools


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
        setting["verbose"] = self.opt.verbose_process
        setting["print_progress"] = self.opt.debug
        setting["gpu_id"] = self.opt.gpu_id

    def list_to_dict(self, space):
        setting = dict(zip(self.get_spaces().keys(), space))
        setting["cuda"] = True
        setting["validate"] = True
        setting["depth"] = 4
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

        if (setting.get("covariance_penalty", 0.) != 0. and
                setting.get("normalized_activations", "NONE") != "NONE"):
            return False

        for level in ["projections", "activations"]:
            rescaled_key = "rescaled_%s" % level
            normalized_key = "normalized_%s" % level
            centered_key = "centered_%s" % level

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
