import glob

dirs = glob.glob('../output_*') #'../simulations_*/*/output_*')
for i, dir in enumerate(dirs):
    spk = glob.glob('%s/spk.dat' % (dir))
    #print i, spk
    if (len(spk) == 0):
        print dir
        #if ('Touch' not in dir):
        #    print dir


