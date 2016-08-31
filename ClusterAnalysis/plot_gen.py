# -*- coding: utf-8 -*-
'''
Plotting results / V-test analysis
Author : Diviyan Kalainathan
Date : 4/08/2016
'''

import csv, numpy, heapq, os
from matplotlib import pyplot as plt

output_folder = 'Cluster_8_obj'

var_to_analyze = [('naf17', 17), ('tranchre', 14), ('statut', 10), ('typemploi', 7), ('csei', 18), ('diplome', 9),
                  ('public', 2), ('public1', 4), ('public2', 4), ('tension1', 2), ('tension2', 2), ('tension3', 2),
                  ('tension4', 2),
                  ('who', 1), ('nbsal', 5), ('sexe', 2)]

inputfile = 'output/' + output_folder + '/numpy-v-test.csv'

permutation_obj = [0, 4, 6, 2, 3, 5, 1, 7]
permutation_subj = [2, 3, 0, 5, 1, 4]
if output_folder[-4:] == 'subj':
    objective = False
    permutation_clusters = permutation_subj
    legend = ['RAS', 'Stress', 'Indep.', 'Heur.', 'Malh.', 'Chgts']
else:
    objective = True
    permutation_clusters = permutation_obj
    legend = ['Indep.', u'Santé', 'Ouvriers', u'CSP+Privé', 'ServPart', 'CSP+Public', 'Immigr.', 'Accid.']

legend = [legend[elt] for elt in permutation_clusters]

mode = 4
# 1 : matrices of v-tests for some vars
# 2 : highest values of v-test per cluster
# 3 : matrices of distance between objective and subjective clusters
# 4 : parallel coordinates of cluster centers
if mode == 1:
    inputdata = numpy.loadtxt(inputfile, delimiter=';')
    n_clusters = inputdata.shape[1]
    for var in var_to_analyze:
        with open('input/n_prepared_data.csv', 'rb') as datafile:
            var_reader = csv.reader(datafile, delimiter=';')
            header = next(var_reader)
            print(len(header))
        name_var = var[0]
        num_var = var[1]
        print(name_var),
        v_test_matrix = numpy.zeros((num_var, n_clusters))
        print(v_test_matrix.shape)
        var_names = []
        row = 0
        for i in [i for i, x in enumerate(header) if (
                            x[0:len(name_var)] == name_var and (
                                    x[0:len(name_var) + 1] == name_var + '_' or num_var == 1) and x[
                                                                                                  -4:] != 'flag')]:
            v_test_matrix[row, :] = inputdata[i, :]
            var_names += [header[i]]
            idx = i
            row += 1
        if name_var == 'diplome':
            print(var_names)

        if name_var == 'naf17':
            var_names = ['Agriculture, sylviculture et peche', 'Fabrication de denrees alimentaires'
                , 'Cokefaction et raffinage', 'Fabrication d\'equipements electriques, electroniques',
                         'Fabrication de materiels de transport',
                         'Fabrication d\'autres produits industriels',
                         'Industries extractives, energie, eau, gestion des dechets',
                         'Construction',
                         'Commerce ; reparation d\'automobiles et de motocycles',
                         'Transports et entreposage',
                         'Hebergement et restauration',
                         'Information et communication',
                         'Activites financieres et d\'assurance',
                         'Activites immobilieres',
                         'Activites scientifiques ; services administratifs et de soutien',
                         'Administration publique, enseignement, sante humaine, action sociale',
                         'Autres activites de services']

        elif name_var == 'csei':
            var_names = ['Agriculteurs exploitants',
                         'Artisans',
                         'Commercants et assimiles',
                         'Chefs d\'entreprise de 10 salaries ou plus',
                         'Professions liberales',
                         'Cadres de la fonction publique, professions intellectuelles et artistiques',
                         'Cadres d\'entreprise',
                         'Professions intermediaires de l\'enseignement, de la sante, de la fonction publique et assimiles',
                         'Professions intermediaires administratives et commerciales des entreprises',
                         'Techniciens',
                         'Contremaitres, agents de maitrise',
                         'Employes de la fonction publique',
                         'Employes administratifs d\'entreprise',
                         'Employes de commerce',
                         'Personnels des services directs aux particuliers',
                         'Ouvriers qualifies',
                         'Ouvriers non qualifies',
                         'Ouvriers agricoles']

        elif name_var == 'diplome':
            var_names = ['0. Aucun diplome',
                         '1. CEP (certificat d\'etudes primaires)',
                         '2. Brevet des colleges, BEPC, brevet elementaire',
                         '3. CAP, BEP ou diplome de ce niveau',
                         '4. Baccalaureat technologique ou professionnel ou diplome de ce niveau',
                         '5. Baccalaureat general ',
                         '6. Diplome de niveau Bac+2',
                         '7. Diplome de niveau bac +3 ou bac +4',
                         '8. Diplome de niveau superieur a bac+4']

        elif name_var == 'statut':
            var_names = ['1. Salarie de l\'etat ',
                         '2. Salarie d\'une collectivite territoriale ',
                         '3. Salarie d\'un hopital public',
                         '4. Salarie d\'un etablissement de sante prive (a but lucratif ou non lucratif)',
                         '5. Salarie du secteur public social et medico-social ',
                         '6. Salarie d\'une entreprise, d\'un artisan, d\'une association ',
                         '7. Salarie d\'un ou plusieurs particuliers',
                         '8. Vous aidez un membre de votre famille dans son travail sans etre remunere',
                         '9. Chef d\'entreprise salarie, PDG, gerant minoritaire, associe',
                         '10. Independant ou a votre compte']
        elif name_var == 'typemploi':
            var_names = ['1. Contrat d\'apprentissage ou de professionnalisation',
                         '2. Placement par une agence d\'interim',
                         '3. Stage remunere en entreprise',
                         '4. Emploi aide ',
                         '5. Autre emploi a duree limitee, CDD, contrat court, saisonnier, vacataire, etc.',
                         '6. Emploi sans limite de duree, CDI, titulaire de la fonction publique',
                         '7. Travail sans contrat']

        elif name_var == 'sexe':
            var_names = ['Homme', 'Femme']

        elif name_var == 'who':
            for n in range(n_clusters):
                print('cluster : ' + str(n))

                n_cluster_data = numpy.loadtxt('output/' + output_folder + '/cluster_' + str(n) + '.csv', delimiter=';')

                v_test_matrix[:, n] = numpy.mean(n_cluster_data[:, idx])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Applying permutations
        v_test_matrix[:, range(v_test_matrix.shape[1])] = v_test_matrix[:, permutation_clusters]
        if name_var != 'who':
            draw = ax.matshow(v_test_matrix, vmin=-15, vmax=15)
        else:
            draw = ax.matshow(v_test_matrix)

        # PCM = ax.get_children()[2]

        # fig, ax1 = plt.subplots()
        # fig=plt.pcolor(v_test_matrix, vmin=-20, vmax=20,linestyle=':')
        plt.title('v-test : ' + name_var, y=1.08)
        plt.ylabel(name_var)
        plt.yticks(xrange(len(var_names)), var_names, rotation='0')
        # x0, x1 = ax1.get_xlim()
        # y0, y1 = ax1.get_ylim()
        # ax1.set_aspect(abs(x1 - x0) / abs(y1 - y0))
        fig.subplots_adjust(right=1, top=0.88, left=0.15, bottom=0.18)
        cbar_ax = fig.add_axes()
        plt.colorbar(draw, cax=cbar_ax)
        plt.xticks(range(len(legend)), legend, rotation=80)
        ax.xaxis.set_ticks_position('bottom')
        # fig.colorbar(ax)
        # plt.tight_layout()
        plt.show()
        # plt.savefig('output/'+output_folder+'/matrix-vt-'+name_var+'.pdf')

    # Computing autonomy & motivation values
    idx = []
    autonomynames = []
    autonomyvars = ['comment_2', 'stark_2', 'stark_3', 'stark_4', 'incident_1', 'incident_2', 'repete_2',
                    'comment_flag', 'stark_flag', 'incident_flag', 'repete_flag']
    a_weights = [3, 1, 2, 3, 3, 1, 2]
    autonomy_matrix = numpy.zeros((n_clusters, 1))
    for a_var in autonomyvars:
        for h in [h for h, x in enumerate(header) if x == a_var]:
            idx += [h]

    for n in range(n_clusters):
        print('cluster : ' + str(n))

        n_cluster_data = numpy.loadtxt('output/' + output_folder + '/cluster_' + str(n) + '.csv', delimiter=';')

        extracted_data = n_cluster_data[:, idx]

        num_valid_values = 0
        for pop_cluster in range(extracted_data.shape[0]):
            if sum(extracted_data[pop_cluster, 7:11]) == 4:
                num_valid_values += 1

        print(num_valid_values)
        autonomy_matrix[n] = sum(sum((numpy.transpose(extracted_data[:, 0:7] * a_weights)) *
                                     (extracted_data[:, 7] * extracted_data[:, 8] * extracted_data[:,
                                                                                    9] * extracted_data[:, 10]))) / (
                                 num_valid_values)

    autonomy_matrix[range(autonomy_matrix.shape[0]), :] = autonomy_matrix[permutation_clusters, :]

    print(autonomy_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw = ax.matshow(autonomy_matrix.transpose())  # , vmin=-15, vmax=15)

    plt.title('Niveau d\'autonomie', y=1.10)
    plt.ylabel('Score d\'autonomie')
    plt.xticks(range(len(legend)), legend, rotation=70)

    fig.subplots_adjust(right=1, top=0.88, left=0.15)
    cbar_ax = fig.add_axes()
    fig.colorbar(draw, cax=cbar_ax)
    ax.xaxis.set_ticks_position('bottom')

    plt.show()


elif mode == 2:
    for var in var_to_analyze:
        with open('input/prepared_data.csv', 'rb') as datafile:
            var_reader = csv.reader(datafile, delimiter=';')
            header = next(var_reader)

    data = numpy.loadtxt(inputfile, delimiter=';')

    with open('output/' + output_folder + '/max-min_v-test.csv', 'wb') as outputfile:
        datawriter = csv.writer(outputfile, delimiter=';')
        datawriter.writerow(['V-tests'])
    max_idx = [[] for i in range(data.shape[1])]
    min_idx = [[] for i in range(data.shape[1])]

    for col in range(data.shape[1]):
        max_idx[col] = heapq.nlargest(20, range(len(data[:, col])), data[:, col].take)
        min_idx[col] = heapq.nsmallest(20, range(len(data[:, col])), data[:, col].take)

    with open('output/' + output_folder + '/max-min_v-test.csv', 'a') as outputfile:
        datawriter = csv.writer(outputfile, delimiter=';')
        datawriter.writerow(['Max values'])

        for i in range(20):
            row = []
            for j in range(data.shape[1]):
                idx = max_idx[j][i]
                row += [header[idx]]
            datawriter.writerow(row)

        datawriter.writerow('')
        datawriter.writerow(['Min values'])

        for i in range(20):
            row = []
            for j in range(data.shape[1]):
                idx = (min_idx[j][i])
                row += [header[idx]]
            datawriter.writerow(row)

            # for col in range(len(data)-1):
            # n_data[row,col]=data[row][col+1]

    # for col in range(len(data[1])-1))
    print(header)
elif mode == 3:
    # 2 is objective
    # 1 is subjective

    clustering2 = 'cluster_predictions_c8_n500_r12-obj.csv'
    clustering1 = 'cluster_predictions_c6_n500_r12-subj.csv'

    result_folder = 'output/obj8Xsubj6/'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    clustering_input_1 = numpy.loadtxt('input/' + clustering1, delimiter=';')
    clusters_1 = numpy.asarray(sorted(clustering_input_1, key=lambda x: x[1]))

    clustering_input_2 = numpy.loadtxt('input/' + clustering2, delimiter=';')
    clusters_2 = numpy.asarray(sorted(clustering_input_2, key=lambda x: x[1]))

    n_clusters_1 = (set((clusters_1[:, 0])))
    n_clusters_2 = (set((clusters_2[:, 0])))

    count_matrix = numpy.zeros((len(n_clusters_1), len(n_clusters_2)))
    # 1st result is intersect/union
    # 2nd is intersect/min(card(Cluster_i),card(Cluster_j))
    norm_obj = numpy.zeros((len(n_clusters_1), len(n_clusters_2)))
    norm_subj = numpy.zeros((len(n_clusters_1), len(n_clusters_2)))

    # Counting values
    for ppl in range(len(clusters_1[:, 0])):
        count_matrix[clusters_1[ppl, 0], clusters_2[ppl, 0]] += 1

    for row in range(len(n_clusters_1)):
        for col in range(len(n_clusters_2)):
            norm_obj[row, col] = float(count_matrix[row, col]) \
                                 / (sum(count_matrix[:, col]))

            norm_subj[row, col] = float(count_matrix[row, col]) \
                                  / sum(count_matrix[row, :])

    # xticks = [str(u) for u in range(1, len(n_clusters_2) + 1)]
    # yticks = [str(u) for u in range(1, len(n_clusters_1) + 1)]

    xticks = ['Indep.', u'Santé', 'Ouvriers', u'CSP+Privé', 'ServPart', 'CSP+Public', 'Immigr.', 'Accid.']
    yticks = ['RAS', 'Stress', 'Indep.', 'Heur.', 'Malh.', 'Chgts']
    xticks = [xticks[elt] for elt in permutation_obj]
    yticks = [yticks[elt] for elt in permutation_subj]

    norm_obj[:, range(len(n_clusters_2))] = norm_obj[:, permutation_obj]
    norm_obj[range(len(n_clusters_1)), :] = norm_obj[permutation_subj, :]

    norm_subj[:, range(len(n_clusters_2))] = norm_subj[:, permutation_obj]
    norm_subj[range(len(n_clusters_1)), :] = norm_subj[permutation_subj, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw = ax.matshow(norm_obj, vmax=0.35)
    plt.title('Matrice de croisement des clusters '
              '\nsubjectifs sur les clusters objectifs '
              '\n normalises sur les clusters objectifs')
    plt.xlabel('Clusters objectifs (somme=1)')
    plt.ylabel('Clusters subjectifs')
    plt.xticks(range(len(n_clusters_2)), xticks, rotation=60)
    plt.yticks(range(len(n_clusters_1)), yticks)
    ax.xaxis.set_ticks_position('bottom')

    numpy.savetxt(result_folder + '/mx_norm_obj.csv', norm_obj)

    cbar_ax = fig.add_axes()
    plt.colorbar(draw, cax=cbar_ax)
    plt.show()
    # plt.savefig(result_folder + '/comp_m_union.pdf')



    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw = ax.matshow(norm_subj, vmax=0.35)
    plt.title('Matrice de croisement des clusters'
              ' \nsubjectifs sur les clusters objectifs '
              '\n normalises sur les clusters subjectifs')
    plt.xlabel('Clusters objectifs')
    plt.ylabel('Clusters subjectifs(somme=1)')
    plt.xticks(range(len(n_clusters_2)), xticks, rotation=60)
    plt.yticks(range(len(n_clusters_1)), yticks)
    ax.xaxis.set_ticks_position('bottom')

    numpy.savetxt(result_folder + '/mx_norm_subj.csv', norm_subj)
    cbar_ax = fig.add_axes()
    plt.colorbar(draw, cax=cbar_ax)
    plt.show()
    # plt.savefig(result_folder + '/comp_m_min.pdf')

elif mode == 4:
    # Parallel coordinates
    clusters_centers_data = numpy.loadtxt('input/cluster_centers_c6_n500_r12-subj.csv', delimiter=';')
    objective = False

    objective_axis_names = [u'Taille d\'entreprise',
                            u'Niveau de \n qualification du travail',
                            u'Temps de travail\n et securité',
                            u'Secteur Privé /\n public',
                            u'Lien à \n l\'immigration',
                            u"Accidents du travail",
                            u"Ancienneté et taille \n du foyer",
                            u"Situation \n familiale"]
    subjective_axis_names = [u"Risques \npsychosociaux",
                             u"Indépendance",
                             u"Bon management",
                             u"Changement du \nmilieu de travail",
                             u"Satisfaction du \n travail en équipe"]

    print(len(objective_axis_names))
    print(objective_axis_names[2])

    if objective:

        permutation_clusters = permutation_obj  # [0,4,6,2,3,5,1,7]
        permutation_axis = [0, 1, 2, 3, 4, 5, 6, 7]
        inversion_axis = [1, -1, 1, 1, -1, -1, -1, 1]
        legend = ['Indep.', u'Santé', 'Ouvriers', u'CSP+Privé', 'ServPart', 'CSP+Public', 'Immigr.', 'Accid.']

        # creating ticks
        xticks = []
        for tick in range(len(objective_axis_names)):
            xticks.append('Axe ' + str(tick + 1) + '\n' + objective_axis_names[tick])
    else:
        permutation_clusters = permutation_subj
        permutation_axis = [0, 1, 3, 2, 4]
        inversion_axis = [1, -1, 1, 1, 1]
        legend = ['RAS', 'Stress', 'Indep.', 'Heur.', 'Malh.', 'Chgts']

        # creating ticks
        xticks = []
        for tick in range(len(subjective_axis_names)):
            xticks.append('Axe ' + str(tick + 1) + '\n' + subjective_axis_names[tick])

    # applying inversions

    clusters_centers_data = clusters_centers_data * inversion_axis
    for factor in range(len(inversion_axis)):
        if inversion_axis[factor] == -1:
            xticks[factor] = '- ' + xticks[factor]

    # applying permutations:

    clusters_centers_data[range(clusters_centers_data.shape[0]), :] = clusters_centers_data[permutation_clusters, :]
    clusters_centers_data[:, range(clusters_centers_data.shape[1])] = clusters_centers_data[:, permutation_axis]

    xticks = [xticks[elt] for elt in permutation_axis]
    legend = [legend[elt] for elt in permutation_clusters]

    for line in range(clusters_centers_data.shape[0]):
        if line < 7:
            plt.plot(range(clusters_centers_data.shape[1]), clusters_centers_data[line], linewidth=2.5)
        else:
            plt.plot(range(clusters_centers_data.shape[1]), clusters_centers_data[line], '--', linewidth=2.5)

    plt.xlabel('Axes')
    plt.xticks(xrange(clusters_centers_data.shape[1]), xticks)
    print(xticks)
    if objective:
        plt.title(u'Représentation en coordonnées parallèles \n des centres des clusters objectifs')
    else:
        plt.title(u'Représentation en coordonnées parallèles \n des centres des clusters subjectifs')
    plt.legend(legend, loc=1)
    plt.grid()
    plt.show()

print('Done !')
