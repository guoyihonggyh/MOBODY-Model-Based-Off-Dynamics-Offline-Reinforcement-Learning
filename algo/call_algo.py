# import all algorithms this benchmark implement

def call_algo(algo_name, config, mode, device, **kwargs):

    algo_name = algo_name.lower()


    from offline_offline.dara import DARA
    from offline_offline.bosa import BOSA
    from offline_offline.iql import IQL
    from offline_offline.td3_bc import TD3BC
    from offline_offline.igdf import IGDF
    from algo.offline_offline.mobody import MOBODY

    algo_to_call = {
        'dara': DARA,
        'bosa': BOSA,
        'iql': IQL,
        'td3_bc': TD3BC,
        'igdf': IGDF,
        'mobody': MOBODY

    }

    algo = algo_to_call[algo_name]
    policy = algo(config, device)
    
    return policy