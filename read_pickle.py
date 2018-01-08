import pickle
a,b,c=pickle.load(open('./assets/learned_models/Hopper-v1_ppo.p', "rb"))
print(a)
print(b)
print(c)
