from feature_generation import distance_and_angle_generator, hydrophobicity_generator, \
    physicochemical_features_generator, ProtBERT_feature_generator, \
    ProtXLNet_feature_generator, residue_accessibility_generator, label_generator

from config import DefaultConfig
configs = DefaultConfig()

def generate_all_features():
    dataset_name = configs.dataset_name
    if dataset_name == 'masif':
        binding_type = 'b'
    else:
        binding_type = 'u'
    distance_and_angle_generator.generate_distance_and_angle_matrix(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('dist and angle l done')
    distance_and_angle_generator.generate_distance_and_angle_matrix(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('dist and angle r done')

    hydrophobicity_generator.generate_hydrophobicity(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('hydro l done')
    hydrophobicity_generator.generate_hydrophobicity(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('hydro r done')
    physicochemical_features_generator.generate_physicochemical_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('phychem l done')
    physicochemical_features_generator.generate_physicochemical_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('phychem r done')

    ProtBERT_feature_generator.generate_protbert_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('protbert l done')
    ProtBERT_feature_generator.generate_protbert_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('protbert r done')
    ProtXLNet_feature_generator.generate_protxlnet_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('protxlnet l done')
    ProtXLNet_feature_generator.generate_protxlnet_features(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('protxlnet r done')
    residue_accessibility_generator.generate_residue_accessibility(input_dir='./inputs/', dataset_name=dataset_name, protein_type='l', binding_type=binding_type)
    print('res acc l done')
    residue_accessibility_generator.generate_residue_accessibility(input_dir='./inputs/', dataset_name=dataset_name, protein_type='r', binding_type=binding_type)
    print('res acc r done')
    label_generator.generate_labels(input_dir='./inputs/', dataset_name=dataset_name, binding_type=binding_type)
    print('label done')

if __name__ == '__main__':
    generate_all_features()