def test_imports():
    import connalysis
    import generate_model
    from generate_model import ER
    from connalysis import modelling
    from connalysis import randomization
    from connalysis.randomization import run_ER
    from connalysis.randomization import adjusted_ER
    from connalysis.modelling import run_batch_model_building
    from connalysis.modelling import run_model_building
    from connalysis.modelling import run_pathway_model_building
    from connalysis.modelling import conn_prob_model
    from connalysis.modelling import conn_prob_pathway_model
    from connalysis.modelling import conn_prob_2nd_order_model
    from connalysis.modelling import conn_prob_2nd_order_pathway_model
    from connalysis.modelling import conn_prob_3rd_order_model
    from connalysis.modelling import conn_prob_3rd_order_pathway_model
    from connalysis import network
    #TODO: commented out until pipy pyflagser distribution is fixed
    import pyflagsercount