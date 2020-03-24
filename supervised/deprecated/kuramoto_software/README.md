# Kuramoto Dynamics Software

Integration of kuramoto dynamics tools, including dynamics evolution, visualization, loss functions and networks

## Getting Started

`kuramoto.py` provides both *pytorch* and *numpy* kuramoto functions. After you instantiate a class, the basic function is `evolution`, you can use `show=True` to show kuramoto running progress, or `record=True` to record the phases and frequencies in progress. You can also set up the default parameters which is announced before in the front of the file. Another thing needs to be noticed is use `phase_init()` function during training to get the phases initialized.


`kura_visual.py` provides animation of kuramoto dynamics. You need to give the phases and frequencies records to the class, and when you use  `animate_evol` or `animate_evol_compare`, specify the image size by providing width and height. The difference between 2 animation functions is basically if there is a mask provided or say if there is a comparison. There are also functions for printing out phases evolution and properties evolution in a PNG image

`loss_func_ex.py` provides various properties measuring the status of the dynamics. Basically includes 'coherence', 'pair-wise inner products' and 'frame potential', both in *numpy* and *pytorch*. Some functions need a input with both phases and masks to, but the important thing is phases are in the size of *(batch, oscillator_number)*, while masks are in the size of *(batch, groups, oscillator_number)*

`net.py` provides networks functions, I haven't written much for this, you can add your own networks in it. But there is a `mask_pro` where you basically provide a dictionary including 1-d index of the special oscillators in one object and it will give you a group of masks (however does not surport batch, use `map` function to solve this problem)

`how_to_use.py` provides a training demo to show how to integrate these files to save dynamics GIF during training and print out what you need which feed in only one exmaple which is saved in `train-data.npy`
