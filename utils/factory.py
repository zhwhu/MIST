def get_model(model_name, args):
    name = model_name.lower()
    if name=="simplecil_mist":
        from models.simplecil_mist import Learner
        return Learner(args)
    elif name=="ranpac_mist":
        from models.ranpac_mist import Learner
        return Learner(args)
    else:
        assert 0
