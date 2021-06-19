"""Modules pour manipuler les images
et représenter nos résultats.
"""
from math import sqrt
import numpy as np
import time
import matplotlib.pyplot as plt


#LES FONCTIONS

def im_distance(A,B):
    """Calcule la distance entre les images A et B."""

    distance = 0
    n = len(A)

    for i in range(n):
        for j in range(n):
            distance += abs(A[i][j] - B[i][j])

    return distance


def gris(im):
    """Renvoie une image qui correspond à im grisée."""

    im_gris = np.empty([len(im),len(im[0])])

    for i in range(len(im)):
        for j in range(len(im[0])):
            im_gris[i][j] = round(np.mean(im[i][j]))

    return im_gris


def somme_liste(L):
    """Permet d'obtenir un élément supérieur au maximum de L."""

    somme = 0

    for el in L:
        somme += el

    return somme


def etiquette_majoritaire(X,indices_min):
    """Renvoie l'étiquette majoritaire parmis les images
    dont les indices sont dans la liste indices_min."""

    nb_pistol,nb_knife = 0,0
    etiquettes = []

    for indice in indices_min:
        etiquettes.append(X[indice][1])

    for el in etiquettes:
        if el == "pistol":
            nb_pistol += 1
        if el == "knife":
            nb_knife += 1

    if nb_pistol > nb_knife:
        return "pistol"
    if nb_knife > nb_pistol:
        return "knife"

    #si il n'y a pas d'etiquette majoritaire,
    #on ignore celle de l'image la plus éloignée
    return etiquette_majoritaire(X, indices_min[:-1])



def Im_kNN(X,A,k):
    """Programme des k plus proches voisins
    X: liste de couples (image,étiquette)
    A: nouvelle image"""

    liste_distances = len(X)*[0]

    for i in range(len(X)):
        liste_distances[i] = im_distance(X[i][0],A)

    Somme = somme_liste(liste_distances)
    indices_min = []

    for _ in range(k):
        Index_min = liste_distances.index(min(liste_distances))
        indices_min.append(Index_min)
        liste_distances[Index_min] = Somme

    return etiquette_majoritaire(X, indices_min)


def Test_kNN(X,x,k):
    """teste sur plusieur images les capacités prédictives de Im_kNN()
    x: liste couple (nouvelle image, etiquette)"""

    n = len(x)
    S = 0

    for i in range(n):
        if x[i][1] == Im_kNN(X,x[i][0],k):
            S += 1

    return S/n


def negatif_image(Im):
    """Renvoie l'image en négatif."""

    n,N = len(Im),len(Im[0])
    negatif = np.empty([n,N])

    for i in range(n):
        for j in range(N):
            negatif[i][j] = 255 - Im[i][j]

    return negatif


def mauvaise_prediction(X,x,k):
    """Renvoie les indices des images ayant eu une mauvaise prédiction."""

    n = len(x)
    images = []

    for i in range(n):
        if x[i][1] != Im_kNN(X,x[i][0],k):
            images.append(i)

    return images


def copier_liste(L):
    """Va permettre de modifier une liste
    tout en préservant l'originale."""

    L2 = []

    for el in L:
        L2.append(el)

    return L2


def distances_mutuelles(X):
    """Renvoie la liste des distances entre les éléments de X
    et X privé de celui-ci.
    Va servir à définir un seuil."""

    X_copy = copier_liste(X)
    distances = []

    for i in range(len(X)-1):
        A = X_copy[i]
        X_copy = X_copy[:i] + X_copy[i+1:]
        distances_temp = []

        for j in range(i,len(X)-1):
            distances_temp.append(im_distance(A[0],X_copy[j][0]))

        distances.append(min(distances_temp))
        X_copy = X_copy[:i] + [A] + X_copy[i:]

    return sorted(distances)


def creer_X():
    """Crée une liste d'intervalles"""

    pas = 5
    X = []

    for i in range(0,100,pas):
        if i%1 == 0:
            X.append(str(i) + "x10**4 - " + str(i + pas) + "x10**4")
        else:
            X.append("")

    return X


def compter_presence(L,x,y):
    """Renvoie le pourcentage d'éléments de L appartenant à ]x,y["""

    S = 0

    for el in L:
        if x < el < y:
            S+=1

    return S/len(L)


def creer_Y(L,pas):
    """Applique comper_presence() aux intervalles obtenus par creer_X()"""

    Y = []

    for i in range(20):
        Y.append(compter_presence(L,i*pas,i*pas+pas))

    return Y


def moyenne_liste(L):
    """Rend nos graphiques plus informatifs"""

    n = len(L)
    S = 0

    for el in L:
        S += el

    return S//n


def distance_minimal(X,A):
    """Renvoie la distance entre l'ensemble des images de X
    et l'image A"""

    liste_distances = len(X)*[0]

    for i in range(len(X)):
        liste_distances[i] = im_distance(X[i][0],A)

    Index_min = liste_distances.index(min(liste_distances))

    return liste_distances[Index_min]


def detection_seuil(X,A,SEUIL):
    """Vérifie si la distance entre A et X est inférieur à un seuil """

    return distance_minimal(X,A) < SEUIL


def ecart_type(L,moyenne):
    """Rend nos graphiques plus informatifs"""

    S = 0

    for el in L:
        S += (el - moyenne)**2

    return sqrt(S/len(L))


def seuil_pourcents(L,p,pas):
    """L: liste triée, p: le pourcentage souhaité
    renvoie le seuil tel que p% des éléments de L soient supérieur"""

    Max = L[-1]
    seuil = Max

    while compter_presence(L,seuil,Max) < p:
        seuil -= pas

    return seuil


def test_detection(X,x,SEUIL):
    """test notre modèle de détection sur plusieurs images"""

    n = len(x)
    S = 0

    for i in range(n):
        if detection_seuil(X,x[i][0],SEUIL):
            S += 1

    return S/n


#L'APPLICATION

#code non exaustif, il permet seulement de donner une idée de comment
#les fonctions ont été utilisées
#les lignes du type "Restultats=..." permettent de stocker ce qu'a renvoyé une fonction

start = time.time()

#Importation des images
Pistol = []
for i in range(695):
    Pistol.append(plt.imread(r'C:\Users\rapha\Desktop\OD-WeaponDetection-master\Pistol_classification\AAAPistol\pistol ('+str(i)+').jpg'))

Knife = []
for i in range(1,487):
    Knife.append(plt.imread(r'C:\Users\rapha\Desktop\OD-WeaponDetection-master\Knife_classification\AAAKnife\knife ('+str(i)+').jpg'))

DATA = Pistol + Knife

DATA_gray = []
for i in range(1181):
    DATA_gray.append(gris(DATA[i]))

DATA_etiquette = []
for i in range(695):
    DATA_etiquette.append((DATA_gray[i],'pistol'))
for i in range(695,1181):
    DATA_etiquette.append((DATA_gray[i],'knife'))


Pistol_test = DATA_etiquette[595:695]
Knife_test = DATA_etiquette[1081:1181]
DATA_train = DATA_etiquette[0:595] + DATA_etiquette[695:1081]

print(time.time() - start," secondes")


#tests avec les images en négatifs

Pistol_test_negatif, Knife_test_negatif = [],[]

for el in Pistol_test:
    Pistol_test_negatif.append((negatif_image(el[0]),el[1]))

for el in Knife_test:
    Knife_test_negatif.append((negatif_image(el[0]),el[1]))

print(Test_kNN(DATA_train,Pistol_test_negatif,3))
print(Test_kNN(DATA_train,Knife_test_negatif,3))


#création du graphique F(nb_images)=perfomance

X = []
for i in range(1,10):
    X.append(i*100)
X.append(981)

Resultats = []

for i in range(1,10):
    print("[0,",100*i,"]")
    resultat_pistol = Test_kNN(DATA_train[:100*i],Pistol_test,1)
    resultat_knife = Test_kNN(DATA_train[:100*i],Knife_test,1)
    Resultats.append((resultat_pistol + resultat_knife)/2)

resultat_pistol = Test_kNN(DATA_train,Pistol_test,1)
resultat_knife = Test_kNN(DATA_train,Knife_test,1)
Resultats.append((resultat_pistol + resultat_knife)/2)


Resultats = [0.5, 0.5, 0.5, 0.5, 0.5, 0.65, 0.7749999999999999, 0.85, 0.88, 0.88]


#création du graphique F(k)=perfomance

Resultats = []

for k in range(1,21):
    restultat_pistol = Test_kNN(DATA_train,Pistol_test,k)
    restultat_knife = Test_kNN(DATA_train,Knife_test,k)

    resultat = (restultat_pistol + restultat_knife)/2

    print("resutltat pour k =",k,"= ",resultat)
    Resultats.append(resultat)



Resultats = [0.88,0.88,0.885,0.885,0.88,0.88,0.8999999999999999,0.8999999999999999,0.885,0.885,0.8999999999999999,0.8999999999999999,0.905,0.905,0.905,0.905]


#exemple création graphique

plt.title("Exactitude en fonction du nombre d'images d'entrainement")
plt.grid(True)
plt.xlabel('nb images')
plt.ylabel('%')
plt.axis([50, 1000, 0, 1])
plt.plot(X,Resultats)


#récupération des données pour les graphiques des distances mutuelles

distances_mutuelles_pistol = distances_mutuelles(DATA_train[0:595])

distances_mutuelles_pistol = [116814.0, 122057.0, 134649.0, 144915.0, 147011.0, 151376.0, 159062.0, 162368.0, 163933.0, 164864.0, 165296.0, 166953.0, 167769.0, 168904.0, 170268.0, 170760.0, 172317.0, 173090.0, 173819.0, 174357.0, 174643.0, 176829.0, 177613.0, 179503.0, 179900.0, 186421.0, 186891.0, 187224.0, 187656.0, 188900.0, 189496.0, 191578.0, 194116.0, 194388.0, 194756.0, 196011.0, 198807.0, 198901.0, 199406.0, 201455.0, 201836.0, 203902.0, 206487.0, 209014.0, 210014.0, 210090.0, 210622.0, 210909.0, 211718.0, 212609.0, 214991.0, 215523.0, 215699.0, 216166.0, 216199.0, 217130.0, 217309.0, 217620.0, 217751.0, 217896.0, 219584.0, 221308.0, 221617.0, 221926.0, 222775.0, 224608.0, 224672.0, 225426.0, 225978.0, 226375.0, 226855.0, 226948.0, 227313.0, 228725.0, 230208.0, 230253.0, 230744.0, 230941.0, 231796.0, 231804.0, 232555.0, 232983.0, 233763.0, 235129.0, 235757.0, 235881.0, 235954.0, 236349.0, 236584.0, 237795.0, 237939.0, 239175.0, 239381.0, 241055.0, 241502.0, 244050.0, 244617.0, 245083.0, 245203.0, 245254.0, 245462.0, 247035.0, 247284.0, 247461.0, 248199.0, 248973.0, 249665.0, 250438.0, 250599.0, 252635.0, 253505.0, 254450.0, 254600.0, 254796.0, 256062.0, 257151.0, 257575.0, 258305.0, 258337.0, 258803.0, 259356.0, 259394.0, 259436.0, 260042.0, 260057.0, 260370.0, 261787.0, 262927.0, 263156.0, 263585.0, 264014.0, 264080.0, 264543.0, 264892.0, 265102.0, 265325.0, 267172.0, 268104.0, 268333.0, 269129.0, 269404.0, 272303.0, 272305.0, 272887.0, 275008.0, 275013.0, 275520.0, 275987.0, 276118.0, 277764.0, 278789.0, 278852.0, 278892.0, 280474.0, 280505.0, 280862.0, 281189.0, 282477.0, 282718.0, 283197.0, 283269.0, 283437.0, 283910.0, 284351.0, 285029.0, 285132.0, 286935.0, 287603.0, 288286.0, 290258.0, 290425.0, 291028.0, 291320.0, 291356.0, 292267.0, 292845.0, 294029.0, 294258.0, 294741.0, 294771.0, 295043.0, 295238.0, 296490.0, 296719.0, 297401.0, 297567.0, 297966.0, 298659.0, 299529.0, 300405.0, 300855.0, 301296.0, 302223.0, 302625.0, 302783.0, 302925.0, 303046.0, 303593.0, 304225.0, 304555.0, 304685.0, 305703.0, 306119.0, 306986.0, 307029.0, 307845.0, 307887.0, 308272.0, 308835.0, 309007.0, 309790.0, 310202.0, 310230.0, 310536.0, 310880.0, 311050.0, 311170.0, 312020.0, 312160.0, 312168.0, 313179.0, 316581.0, 317301.0, 318302.0, 319678.0, 319906.0, 319987.0, 320583.0, 320649.0, 320879.0, 321118.0, 321298.0, 321697.0, 322031.0, 322181.0, 322652.0, 323595.0, 323658.0, 325296.0, 325945.0, 327021.0, 327859.0, 328549.0, 328705.0, 328925.0, 330574.0, 330777.0, 331425.0, 332272.0, 333353.0, 335291.0, 335744.0, 335894.0, 336391.0, 336416.0, 336715.0, 337517.0, 339130.0, 339267.0, 339618.0, 340148.0, 340255.0, 340283.0, 340505.0, 341399.0, 342377.0, 343486.0, 343912.0, 344325.0, 344619.0, 346488.0, 347922.0, 348299.0, 348563.0, 350236.0, 353031.0, 354173.0, 354910.0, 355108.0, 355637.0, 356690.0, 357675.0, 358362.0, 358390.0, 358427.0, 359298.0, 359417.0, 359681.0, 359802.0, 359875.0, 361550.0, 361968.0, 362570.0, 363530.0, 363819.0, 364285.0, 364339.0, 364578.0, 365069.0, 366097.0, 367137.0, 367547.0, 367725.0, 367918.0, 368191.0, 368204.0, 368581.0, 369181.0, 369391.0, 369579.0, 370857.0, 371090.0, 372186.0, 372215.0, 374316.0, 374879.0, 375336.0, 375476.0, 376028.0, 376084.0, 376421.0, 376921.0, 377013.0, 377544.0, 377611.0, 378330.0, 379994.0, 383263.0, 383470.0, 384033.0, 385343.0, 385570.0, 385804.0, 385910.0, 387349.0, 388005.0, 388068.0, 388208.0, 388409.0, 389921.0, 391150.0, 391386.0, 391524.0, 392144.0, 392694.0, 392990.0, 393797.0, 394053.0, 394254.0, 394756.0, 395288.0, 395674.0, 396001.0, 396469.0, 396973.0, 398132.0, 399037.0, 401077.0, 401453.0, 401638.0, 402995.0, 403541.0, 403838.0, 404050.0, 404122.0, 404131.0, 404139.0, 405365.0, 406712.0, 407387.0, 408311.0, 408556.0, 408624.0, 408890.0, 408997.0, 409526.0, 409997.0, 410596.0, 411767.0, 412981.0, 413346.0, 413566.0, 413902.0, 414449.0, 416306.0, 416940.0, 417129.0, 418025.0, 419108.0, 419570.0, 420713.0, 421960.0, 422835.0, 423292.0, 423817.0, 423859.0, 423965.0, 426046.0, 426194.0, 430283.0, 431029.0, 431221.0, 432048.0, 432259.0, 434246.0, 434980.0, 437069.0, 437208.0, 437812.0, 437952.0, 438706.0, 439811.0, 440324.0, 440599.0, 441805.0, 441903.0, 442010.0, 442430.0, 443958.0, 444379.0, 444755.0, 445281.0, 445302.0, 445992.0, 448544.0, 451094.0, 451245.0, 452023.0, 452455.0, 452535.0, 454994.0, 455491.0, 455679.0, 457853.0, 457903.0, 458236.0, 461927.0, 462840.0, 462929.0, 463404.0, 465040.0, 465704.0, 468146.0, 468636.0, 469170.0, 470637.0, 471198.0, 471631.0, 473521.0, 473562.0, 476215.0, 478202.0, 478627.0, 480197.0, 480523.0, 482154.0, 483939.0, 485950.0, 486045.0, 486220.0, 486415.0, 486988.0, 487427.0, 489403.0, 490307.0, 492307.0, 493852.0, 494827.0, 495491.0, 496374.0, 500214.0, 501050.0, 501245.0, 502878.0, 503869.0, 504515.0, 505531.0, 506756.0, 508147.0, 508150.0, 508994.0, 511233.0, 513158.0, 514607.0, 514630.0, 515207.0, 515962.0, 516799.0, 518417.0, 519538.0, 519548.0, 520412.0, 523308.0, 523980.0, 524395.0, 528365.0, 531560.0, 531701.0, 532419.0, 535199.0, 536155.0, 538092.0, 541778.0, 542595.0, 543694.0, 547052.0, 548912.0, 550774.0, 552681.0, 555608.0, 556688.0, 557058.0, 559330.0, 561712.0, 564481.0, 568561.0, 569582.0, 570249.0, 570923.0, 576349.0, 576576.0, 580821.0, 581744.0, 584859.0, 586641.0, 587423.0, 589475.0, 589556.0, 589778.0, 590245.0, 590594.0, 592866.0, 597426.0, 597862.0, 601630.0, 604627.0, 605836.0, 606100.0, 610177.0, 614637.0, 623065.0, 630856.0, 633962.0, 636738.0, 636868.0, 639517.0, 641761.0, 641865.0, 646167.0, 648934.0, 653781.0, 654342.0, 656090.0, 656099.0, 672796.0, 674718.0, 676404.0, 677959.0, 678815.0, 681066.0, 684480.0, 685346.0, 687163.0, 688750.0, 702101.0, 707965.0, 709496.0, 714573.0, 717895.0, 723375.0, 730801.0, 738400.0, 739067.0, 747836.0, 748265.0, 749428.0, 751551.0, 757105.0, 759626.0, 779683.0, 782288.0, 797029.0, 806257.0, 826925.0, 862501.0, 869316.0, 880403.0, 886524.0, 970232.0, 995555.0, 1010933.0, 1172491.0, 1175023.0, 1186149.0]

distances_mutuelles_knife = [0.0, 0.0, 17785.0, 61831.0, 70164.0, 87443.0, 87783.0, 94652.0, 103761.0, 104520.0, 106376.0, 107306.0, 111688.0, 113951.0, 115193.0, 123062.0, 124536.0, 126948.0, 129551.0, 132516.0, 132719.0, 133310.0, 133356.0, 134720.0, 134883.0, 135648.0, 139914.0, 142141.0, 143309.0, 143944.0, 145962.0, 146585.0, 147487.0, 150792.0, 152533.0, 154399.0, 156789.0, 156919.0, 159848.0, 162705.0, 164613.0, 166967.0, 168343.0, 169805.0, 170722.0, 172229.0, 175211.0, 175283.0, 175825.0, 176394.0, 179992.0, 179997.0, 180521.0, 180624.0, 182959.0, 184563.0, 186307.0, 189199.0, 191328.0, 193678.0, 194145.0, 196071.0, 198505.0, 200907.0, 201841.0, 203479.0, 204412.0, 206105.0, 206895.0, 208828.0, 208853.0, 209344.0, 209450.0, 210723.0, 211500.0, 211918.0, 212443.0, 212611.0, 213415.0, 213465.0, 213506.0, 215268.0, 215763.0, 220524.0, 221078.0, 221150.0, 222346.0, 222628.0, 224976.0, 231564.0, 232836.0, 233664.0, 240711.0, 241326.0, 246725.0, 247947.0, 250755.0, 252753.0, 253567.0, 255079.0, 257449.0, 258467.0, 259395.0, 259705.0, 260348.0, 261653.0, 264964.0, 265622.0, 267080.0, 268534.0, 273080.0, 274552.0, 275473.0, 278037.0, 278678.0, 279099.0, 283905.0, 285336.0, 295319.0, 300736.0, 318209.0, 321904.0, 322535.0, 325240.0, 327943.0, 333041.0, 333400.0, 333891.0, 336345.0, 336607.0, 336779.0, 342537.0, 352425.0, 354273.0, 356096.0, 356836.0, 360009.0, 365461.0, 368639.0, 373805.0, 374448.0, 375167.0, 378897.0, 382116.0, 385758.0, 389155.0, 391333.0, 396081.0, 396250.0, 398280.0, 403852.0, 406292.0, 407260.0, 409792.0, 411812.0, 412950.0, 414309.0, 416044.0, 420042.0, 426513.0, 427267.0, 427863.0, 428840.0, 434065.0, 435876.0, 440074.0, 441388.0, 441879.0, 442070.0, 443677.0, 444684.0, 446712.0, 453012.0, 458585.0, 461370.0, 462797.0, 468406.0, 469237.0, 469870.0, 478095.0, 483616.0, 487725.0, 488380.0, 490509.0, 491651.0, 492342.0, 495184.0, 496722.0, 498821.0, 500673.0, 500890.0, 501091.0, 506732.0, 507498.0, 508050.0, 519158.0, 523714.0, 525830.0, 526625.0, 537638.0, 543398.0, 544559.0, 544741.0, 546295.0, 549406.0, 551542.0, 556505.0, 558441.0, 559702.0, 560689.0, 567718.0, 571395.0, 573661.0, 581288.0, 591644.0, 592110.0, 592425.0, 594225.0, 599130.0, 600912.0, 603188.0, 604420.0, 612131.0, 617982.0, 620508.0, 629414.0, 631612.0, 632554.0, 632570.0, 634482.0, 636823.0, 638895.0, 642534.0, 647773.0, 651818.0, 652993.0, 654713.0, 663661.0, 669950.0, 670943.0, 676059.0, 677543.0, 680276.0, 682958.0, 691094.0, 696722.0, 703487.0, 705939.0, 707105.0, 707171.0, 721659.0, 724962.0, 731034.0, 731118.0, 731710.0, 735883.0, 735900.0, 738430.0, 758172.0, 759223.0, 762274.0, 767392.0, 769546.0, 774646.0, 776201.0, 785474.0, 787341.0, 793694.0, 807752.0, 818102.0, 825350.0, 829028.0, 852536.0, 863739.0, 872168.0, 879860.0, 891761.0, 926695.0, 928509.0, 954363.0, 961370.0, 1082676.0, 1167975.0, 1230239.0, 1696491.0]

X = creer_X()
Y = creer_Y(distances_mutuelles_pistol,50000)

X = creer_X()
Y = creer_Y(distances_mutuelles_knife,50000)

plt.bar(X,Y)
plt.show()


#définition du seuil

SEUIL = seuil_pourcents(distances_mutuelles_pistol,0.05,100)


