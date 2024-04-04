from sklearn.mixture import GaussianMixture
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="02")
    parser.add_argument('--output_file', type=str, default="03")
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    prob_names = []
    prob_reps = []
    f = open("result/"+args.input_file+".txt", 'r')
    lines = f.readlines()
    f.close()
    st = 0
    for line in lines:
        if st == 0:
            prob_names.append(line[:-1])
            st = 1
        else:
            rep = [float(x) for x in line.split()]
            prob_reps.append(rep)
            st = 0

    gmm = GaussianMixture(n_components=args.class_num, random_state=args.seed)
    gmm.fit(prob_reps)
    labels = gmm.predict(prob_reps)

    stat = [0 for i in range(args.class_num)]
    for label in labels:
        stat[label] += 1
    print(stat)

    f = open("result/"+args.output_file+".txt", 'w')
    for i in range(len(prob_names)):
        f.write(prob_names[i]+" ")
        f.write(str(labels[i])+"\n")

    f.close()