{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "75c66c50-5902-488e-8902-08048296b5bb",
   "metadata": {},
   "source": [
    "Object Detection  [Classification + Localization]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "572993cf-fb75-4a64-b16d-c03bbc9c27f4",
   "metadata": {},
   "source": [
    "## 1. Importing Libraries "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "edcd67f8-55cb-45d1-92e1-419f98d8b46b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "93c4892a-a2bc-4564-bfed-5373b003da46",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import PIL.Image, PIL.ImageFont, PIL.ImageDraw\n",
    "import tensorflow as tf\n",
    "import tensorflow_datasets as tfds"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "859feb74-8d74-4713-b04d-c2d9b4647ad1",
   "metadata": {},
   "source": [
    "## 2. Visualization Utilities "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8cdb13e5-87f9-45f6-a09b-479d97c842da",
   "metadata": {},
   "outputs": [],
   "source": [
    "im_width =75\n",
    "im_height = 75\n",
    "use_normalized_coordiates = True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8a250f5c-8ce3-4721-b89f-cd470b53fbe2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def draw_bounding_boxes_on_image_array(image,boxes,color=[],thickness=1,display_str_list=()):\n",
    "    image_pil = PIL.Image.fromarray(image)\n",
    "    rgbimg = PIL.Image.new(\"RGBA\", image_pil.size)\n",
    "    rgbimg.paste(image_pil)\n",
    "    draw_bounding_boxes_on_image(rgbimg,boxes,color,thickness,display_str_list)\n",
    "    return np.array(rgbimg)\n",
    "    \n",
    "def draw_bounding_boxes_on_image(image,boxes,color=[],thickness=1,display_str_list=()):\n",
    "    boxes_shape = boxes.shape\n",
    "    if not boxes_shape:\n",
    "        return\n",
    "    if len(boxes_shape) != 2 or boxes_shape[1] != 4:\n",
    "        raise ValueError('Input must be of size [N,4]')\n",
    "    for i in range(boxes_shape[0]):\n",
    "        draw_bounding_box_on_image(image,boxes[i,1],boxes[i,0],boxes[i,3],boxes[i,2],color[i],thickness,display_str_list[i])\n",
    "        \n",
    "def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=1, display_str_list= None, use_normalized_coordiates = True):\n",
    "    draw = PIL.ImageDraw.Draw(image)\n",
    "    im_width,im_height = image.size\n",
    "    if use_normalized_coordiates:\n",
    "        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, \n",
    "                                      ymin * im_height, ymax * im_height)\n",
    "    else:    \n",
    "        (left, right, top, bottom) = (xmin, xmax, ymin , ymax)\n",
    "    draw.line([(left, top),(left, bottom), (right, bottom),\n",
    "               (right, top), (left,top)],width=thickness,fill=color)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "886b1571-7d85-4907-8ff8-772182364ed5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def dataset_to_numpy_util(training_dataset, validation_dataset, N):\n",
    "    batch_train_ds = training_dataset.unbatch().batch(N)\n",
    "\n",
    "    if tf.executing_eagerly():\n",
    "        for validation_digits, (validation_labels, validation_bboxes) in validation_dataset:\n",
    "            validation_digits = validation_digits.numpy()\n",
    "            validation_labels = validation_labels.numpy()\n",
    "            validation_bboxes = validation_bboxes.numpy()\n",
    "            break\n",
    "        for training_digits, (training_labels, training_bboxes) in training_dataset:\n",
    "            training_digits = training_digits.numpy()\n",
    "            training_labels = training_labels.numpy()\n",
    "            training_bboxes = training_bboxes.numpy()\n",
    "            break \n",
    "    validation_labels = np.argmax(validation_labels, axis = 1)\n",
    "    training_labels = np.argmax(training_labels, axis = 1)\n",
    "    return (training_digits, training_labels, training_bboxes,\n",
    "           validation_digits, validation_labels, validation_bboxes)\n",
    "            \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7a468f56-3210-445c-9ee0-342e4d9b3a8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "MATHPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__),\"mpl-data/fonts/ttf\")\n",
    "def create_digits_from_local_fonts(n):\n",
    "    font_labels = []\n",
    "    img = PIL.Image.new('LA',(75*n, 75), color = (0,255))\n",
    "    font1 = PIL.ImageFont.truetype(os.path.join(MATHPLOTLIB_FONT_DIR,'DejaVuSansMono-Oblique.ttf'),25)\n",
    "    font2 = PIL.ImageFont.truetype(os.path.join(MATHPLOTLIB_FONT_DIR,'STIXGeneral.ttf'),25)\n",
    "    d = PIL.ImageDraw.Draw(img)\n",
    "    for i in range(n):\n",
    "        font_labels.append(i%10)\n",
    "        d.text((7+i*75, 0 if i < 10 else -4 ), str(i%10), fill = (255,255), font = font1 if i < 10 else font2)\n",
    "    font_digits = np.array(img.getdata()), np.float32[:,0]/255.0\n",
    "    font_digits = np.reshape(np.stack(np.split(np.reshape(font_digits, [75, 75*n]), n ,axis = 1) ,axis = 0) [n,75*75])\n",
    "    return font_digits, font_lables\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "da51b78b-b4c0-43ca-be54-beb14efd81c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_digits_with_boxes(digits, predictions,labels, pred_bboxes, bboxes, iou, title):\n",
    "\n",
    "    n = 10\n",
    "\n",
    "    indexes = np.random.choice(len(predictions), size = n)\n",
    "    n_digits = digits[indexes]\n",
    "    n_predictions = predictions[indexes]\n",
    "    n_labels = labels[indexes]\n",
    "\n",
    "    n_iou = []\n",
    "    if len(iou) > 0:\n",
    "        n_iou = iou[indexes]\n",
    "\n",
    "    if len(pred_bboxes) > 0:\n",
    "        n_pred_bboxes = pred_bboxes[indexes]\n",
    "    if len(bboxes) > 0:\n",
    "        n_bboxes = bboxes[indexes]\n",
    "    n_digits = n_digits * 255.0\n",
    "    n_digits = n_digits.reshape(n, 75, 75)\n",
    "    fig = plt.figure(figsize = (20,4))\n",
    "    plt.title(title)\n",
    "    plt.xticks([])\n",
    "    plt.yticks([])\n",
    "\n",
    "    for i in range(n):\n",
    "        ax = fig.add_subplot(1,n, i+1)\n",
    "        bboxes_to_plot = []\n",
    "        if (len(pred_bboxes) > i):\n",
    "            bboxes_to_plot.append(n_pred_bboxes[i])\n",
    "            \n",
    "        if (len(bboxes) > i):\n",
    "            bboxes_to_plot.append(n_bboxes[i]) \n",
    "\n",
    "        img_to_draw = draw_bounding_boxes_on_image_array(image=n_digits[i],\n",
    "                                                         boxes=np.asarray(bboxes_to_plot),\n",
    "                                                         color=['red','green'],\n",
    "                                                         display_str_list=[\"True\", 'Pred'])\n",
    "        plt.xlabel(n_predictions[i]) \n",
    "        plt.xticks([])\n",
    "        plt.yticks([])\n",
    "\n",
    "        if n_predictions[i] != n_labels[i]:\n",
    "            ax.axis.label.set_color('red')\n",
    "        plt.imshow(img_to_draw)\n",
    "\n",
    "        if len(iou) > i:\n",
    "            color = \"black\"\n",
    "            if (n_iou[i][0] < iou_threshold):\n",
    "                color = \"red\"\n",
    "            ax.text(0.2, -0.3,\"iou:%s\"%(n_iou[i][0]), color = color, transform = ax.transAxes)   \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c93ba38e-65be-4b19-a453-49fce674cfb0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_metrics(metric_name, title):\n",
    "    plt.title(title)\n",
    "    plt.plot(history.history[metric_name], color = 'blue', label = metric_name)\n",
    "    plt.plot(history.history['val_' + metric_name], color = 'green', label = 'val_' + metric_name)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c6a1272-1496-406f-9a6e-22a72d6d74d8",
   "metadata": {},
   "source": [
    "# 3. Loading and Preprocessing the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "c9552179-b9d9-48bf-b32f-e70e4e4dadda",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "strategy = tf.distribute.get_strategy()\n",
    "strategy.num_replicas_in_sync"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "26bbccdf-ddbb-4818-aa05-99d04c655ba2",
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 64 * strategy.num_replicas_in_sync\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "c483b4a1-4e0f-463e-9704-f21f56bc2138",
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_image_tfds(image, label):\n",
    "    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)\n",
    "    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)\n",
    "    image = tf.reshape(image,(28,28,1,))\n",
    "    image = tf.image.pad_to_bounding_box(image,ymin,xmin, 75, 75)\n",
    "    image = tf.cast(image, tf.float32)/ 255.0\n",
    "    xmin = tf.cast(xmin, tf.float32)\n",
    "    ymin = tf.cast(ymin, tf.float32)\n",
    "\n",
    "    xmax = (xmin + 28)/ 75\n",
    "    ymax = (ymin + 28)/ 75\n",
    "    xmin = xmin / 75\n",
    "    ymin = ymin / 75\n",
    "\n",
    "    return image,(tf.one_hot(label, 10), [xmin, ymin,xmax, ymax])\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ff00583d-2419-4baa-8de3-9cb9c184ed7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_training_dataset():\n",
    "\n",
    "    with strategy.scope():\n",
    "        dataset = tfds.load(\"mnist\", split = \"train\", as_supervised = True, try_gcs = True)\n",
    "        dataset = dataset.map(read_image_tfds, num_parallel_calls = 16)\n",
    "        dataset = dataset.shuffle(5000, reshuffle_each_iteration = True)\n",
    "        dataset = dataset.repeat()\n",
    "        dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)\n",
    "        dataset = dataset.prefetch(-1)\n",
    "    return dataset  \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ae71e803-672b-4bc4-b8b6-0b01845a0ee1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_validation_dataset():\n",
    "\n",
    "    with strategy.scope():\n",
    "        dataset = tfds.load(\"mnist\", split = \"train\", as_supervised = True, try_gcs = True)\n",
    "        dataset = dataset.map(read_image_tfds, num_parallel_calls = 16)\n",
    "        dataset = dataset.batch(10000, drop_remainder = True)\n",
    "        dataset = dataset.repeat()\n",
    "    return dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "31ca8a3c-600a-4985-8d4c-c752e0d0d871",
   "metadata": {},
   "outputs": [],
   "source": [
    "with strategy.scope():\n",
    "    training_dataset = get_training_dataset()\n",
    "    validation_dataset = get_validation_dataset()    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "7d1c7b4a-adac-4b60-8950-498893d025b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "(training_digits, training_labels, training_bboxes,\n",
    " validation_digits, validation_labels, validation_bboxes) = dataset_to_numpy_util(training_dataset, validation_dataset, 10) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ab60279-4617-4533-9b02-a7ab4412e19c",
   "metadata": {},
   "source": [
    "# Visualize Data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "779efafa-93ad-48be-bee4-786af911878e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABiIAAAFcCAYAAABFraaEAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABlBElEQVR4nO3dd3xc5Z3v8e90aSSNeu+WZFvFTbZjmrEJxBAMAVKWJSGENG42ZZeFTQIke5NsCNwkN8lmS0gPgV1CNgkkJEDophjjXuUiW5Kt3utI02fuH4rm4rjJtkbHkj7v10sv8NHM6Dfy8Zlzzvd5fo8pEolEBAAAAAAAAAAAEANmowsAAAAAAAAAAACzF0EEAAAAAAAAAACIGYIIAAAAAAAAAAAQMwQRAAAAAAAAAAAgZggiAAAAAAAAAABAzBBEAAAAAAAAAACAmCGIAAAAAAAAAAAAMUMQAQAAAAAAAAAAYoYgAgAAAAAAAAAAxAxBBAAAAGYMk8k0qa8NGzac18/56le/KpPJdE7P3bBhw5TUcD4/e+LLbrcrMzNTl156qb70pS/p2LFjJzzn4Ycflslk0tGjR8/65x09elQmk0kPP/xwdNubb76pr371qxocHDz3N3ISXq9X9957r0pKShQfH6/y8nL93d/93Vm9xu23367ExMQpqWfid/3b3/52Sl7v7a9pxL4DAAAAxJLV6AIAAACAydq0adNxf/7617+uV155RS+//PJx26uqqs7r53ziE5/QNddcc07Pra2t1aZNm867hvPxwAMP6IorrlAoFFJfX582b96sn//85/re976nn/zkJ/rQhz4Ufez69eu1adMm5ebmnvXPyc3N1aZNm1RWVhbd9uabb+prX/uabr/9dqWkpEzF25EkfeELX9BDDz2k+++/XytXrtShQ4f005/+dMpeHwAAAEDsEEQAAABgxrjooouO+3NmZqbMZvMJ2//a2NiYnE7npH9OQUGBCgoKzqlGl8t1xnpiraKi4rga3vOe9+juu+/WVVddpdtvv12LFy/WokWLJI3/DjMzM8/p5zgcjml7r7/+9a/1vve9T1/84hclSe985zvPekYEAAAAAGPQmgkAAACzytq1a1VTU6PXXntNl1xyiZxOpz72sY9JGr+ZvW7dOuXm5io+Pl6VlZW65557NDo6etxrnKw1U0lJia677jr9+c9/Vm1treLj47Vw4UL9/Oc/P+5xJ2uvM9ES6MiRI7r22muVmJiowsJC3X333fL5fMc9v7W1Ve9///uVlJSklJQUfehDH9LWrVtPaIF0ttLS0vSjH/1IwWBQ3/ve96LbT9aaKRKJ6IEHHlBxcbHi4uK0YsUKvfDCC1q7dq3Wrl0bfdxft2b66le/qs9//vOSpNLS0hNaZb388stau3at0tPTFR8fr6KiIr3vfe/T2NjYGeu3WCw6fPiwIpHIOf8OJuPIkSP66Ec/qoqKCjmdTuXn5+v666/X3r17T/p4r9eru+66Szk5OYqPj9eaNWu0c+fOEx63bds2vec971FaWpri4uK0bNky/c///M8Z62lsbNTf/u3fKi8vTw6HQ9nZ2bryyiu1a9eu832rAAAAwLRhRgQAAABmnY6ODt166636whe+oAceeEBm8/j4m8OHD+vaa6/VnXfeqYSEBB08eFDf/OY3tWXLlhPaO53M7t27dffdd+uee+5Rdna2fvrTn+rjH/+4ysvLdfnll5/2uYFAQO95z3v08Y9/XHfffbdee+01ff3rX1dycrL+9//+35Kk0dFRXXHFFerv79c3v/lNlZeX689//rNuvvnm8/+lSFq5cqVyc3P12muvnfZxX/rSl/Tggw/qjjvu0Hvf+161tLToE5/4hAKBgObPn3/K533iE59Qf3+//v3f/11PPPFEtN1TVVWVjh49qvXr12v16tX6+c9/rpSUFLW1tenPf/6z/H7/GWes3HHHHfra176mz3/+8/q///f/nv2bn6T29nalp6fr//yf/6PMzEz19/frl7/8pVatWqWdO3dqwYIFxz3+vvvuU21trX76059qaGhIX/3qV7V27Vrt3LlT8+bNkyS98soruuaaa7Rq1Sr98Ic/VHJysh5//HHdfPPNGhsb0+23337Keq699lqFQiF961vfUlFRkXp7e/Xmm29O+RocAAAAQCwRRAAAAGDW6e/v129+8xu9853vPG77l7/85ej/RyIRXXrppaqsrNSaNWu0Z88eLV68+LSv29vbq40bN6qoqEiSdPnll+ull17SY489dsYgwu/362tf+5o+8IEPSJKuvPJKbdu2TY899lg0iPjlL3+pI0eO6Nlnn42uUbFu3TqNjY3pRz/60dn9Ek6hqKhIe/bsOeX3BwYG9N3vflc333zzcT+zpqZGF1988WmDiIKCgujvZtmyZSopKYl+74UXXpDX69W3v/1tLVmyJLr9gx/84BlrHhkZiYYA3/nOd+RwOPSNb3zjjM87F5dffvlxf5ehUEjr169XdXW1fvSjH+m73/3ucY/PzMzUk08+GZ1Bc9lll6miokIPPvigfvKTn0iSPv3pT6u6ulovv/yyrNbxS7Crr75avb29uu+++3TbbbdFw7K36+vr06FDh/Sv//qvuvXWW6Pb3/ve9075+wYAAABiidZMAAAAmHVSU1NPCCGk8TY3H/zgB5WTkyOLxSKbzaY1a9ZIkg4cOHDG1126dGn0RrskxcXFaf78+Tp27NgZn2symXT99dcft23x4sXHPffVV19VUlLSCQtl33LLLWd8/ck6U2ujt956Sz6fT3/zN39z3PaLLrrouGDhbC1dulR2u1133HGHfvnLX6qxsXHSz73lllvU3t6u3bt36/7779cDDzwQDW+k8XZWJpNJv/jFL865vgnBYFAPPPCAqqqqZLfbZbVaZbfbdfjw4ZPuIx/84AePa+NVXFysSy65RK+88oqk8VZPBw8ejC4QHgwGo1/XXnutOjo6dOjQoZPWkpaWprKyMn3729/Wd7/7Xe3cuVPhcPi83yMAAAAw3QgiAAAAMOtMtAR6O7fbrdWrV2vz5s26//77tWHDBm3dulVPPPGEJMnj8ZzxddPT00/Y5nA4JvVcp9OpuLi4E57r9Xqjf+7r61N2dvYJzz3ZtnPV3NysvLy8U36/r6/vlD/zfOooKyvTiy++qKysLH3mM59RWVmZysrK9P3vf/+0z9u6dauefvpp3XPPPXI4HPrSl76kBx54QF//+tf1ta99TdL4uhwWi0VXX331Odc34a677tI///M/68Ybb9Qf//hHbd68WVu3btWSJUtO+veck5Nz0m0Tv8euri5J0j/90z/JZrMd9/XpT39a0vhMm5MxmUx66aWXdPXVV+tb3/qWamtrlZmZqb//+7/XyMjIeb9XAAAAYLrQmgkAAACzzl8vNC2NL5Tc3t6uDRs2RGdBSLqgeu2np6dry5YtJ2zv7OycktffsmWLOjs79fGPf/y0NUj//wb6X9dxPrMiVq9erdWrVysUCmnbtm3693//d915553Kzs7W3/7t3570OQ0NDZIkl8sV3XbvvffKbDbrnnvuUTgc1mOPPaaPfexjpw1YJuu//uu/dNttt+mBBx44bntvb69SUlJOePzJ/m46Ozujv8eMjIxozadqqfTX6068XXFxsX72s59Jkurr6/U///M/+upXvyq/368f/vCHk3pPAAAAgNGYEQEAAIA5YSKccDgcx22fqrUXpsKaNWs0MjKiZ5999rjtjz/++Hm/dn9/vz71qU/JZrPpH//xH0/5uFWrVsnhcOjXv/71cdvfeuutSbWgmvj9nm6WiMVi0apVq/Sf//mfkqQdO3ac8rE1NTWSpEceeeS47V/84hf1jW98Q//yL/+i7u5uffvb3z5jbZNhMplO2EeefvpptbW1nfTxv/rVr45rd3Xs2DG9+eabWrt2raTxkKGiokK7d+/WihUrTvqVlJQ0qdrmz5+vL3/5y1q0aNFpf2cAAADAhYYZEQAAAJgTLrnkEqWmpupTn/qUvvKVr8hms+m///u/tXv3bqNLi/rIRz6i733ve7r11lt1//33q7y8XM8++6yee+45STrpgsYnc/jwYb311lsKh8Pq6+vT5s2b9bOf/UzDw8N65JFHVF1dfcrnpqWl6a677tKDDz6o1NRU3XTTTWptbdXXvvY15ebmnrGGRYsWSZK+//3v6yMf+YhsNpsWLFig//7v/9bLL7+s9evXq6ioSF6vVz//+c8lSVddddUpX6+mpkZ/93d/p4ceekjDw8O67bbblJycrLq6Ov30pz9VQUGB2tra9M///M/6t3/7tzP+bkKhkH7729+esD0hIUHvfve7dd111+nhhx/WwoULtXjxYm3fvl3f/va3VVBQcNLX6+7u1k033aRPfvKTGhoa0le+8hXFxcXp3nvvjT7mRz/6kd797nfr6quv1u233678/Hz19/frwIED2rFjh37zm9+c9LX37Nmjz372s/rABz6giooK2e12vfzyy9qzZ4/uueeeM75XAAAA4EJBEAEAAIA5IT09XU8//bTuvvtu3XrrrUpISNANN9ygX//616qtrTW6PEnjN8Nffvll3XnnnfrCF74gk8mkdevW6Qc/+IGuvfbak7YGOpn77rtPkmS1WpWcnKz58+frYx/7mO644w4VFxef8fnf+MY3lJCQoB/+8If6xS9+oYULF+qhhx7Sl770pTPWsHbtWt1777365S9/qZ/85CcKh8N65ZVXtHTpUj3//PP6yle+os7OTiUmJqqmpkZPPfWU1q1bd9rX/M///E+tWLFCP/rRj/Th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",
      "text/plain": [
       "<Figure size 2000x400 with 11 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display_digits_with_boxes(training_digits, training_labels,\n",
    "                          training_labels, np.array([]), training_bboxes,\n",
    "                          np.array([]), \"Training Digits & Labels\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "6d0e5701-3237-41e9-99c4-ad86300ac4cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABiIAAAFcCAYAAABFraaEAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAByrUlEQVR4nO3dd3Rc9Z3+8Wd6n1Hv1XK35Q4uNIfqBAgEsiRASCOFJLsbNtnsJmHTNj3ZXzbZ7ELKCSGBxcDSE0LvGBv3Jhe5qVi9l9H0md8fjiYIG1eNRpLfr3N0OJq5c+dz5cu9M/e538/XkEgkEgIAAAAAAAAAAEgBY7oLAAAAAAAAAAAAkxdBBAAAAAAAAAAASBmCCAAAAAAAAAAAkDIEEQAAAAAAAAAAIGUIIgAAAAAAAAAAQMoQRAAAAAAAAAAAgJQhiAAAAAAAAAAAAClDEAEAAAAAAAAAAFKGIAIAAAAAAAAAAKQMQQQAAADGhQ984ANyOBzq7e1912VuvvlmWSwWtbW1nfR6DQaDvv3tbyd/f+WVV2QwGPTKK6+c8LUf//jHVVFRcdLv9XZ33nmn7rnnnqMer6urk8FgOOZzqfbtb39bBoMh+eN0OlVSUqIrrrhCv/zlLzUwMHDUa87kb3DPPffIYDCorq4u+dj999+vn//856e3Acdx+PBhffjDH1ZeXp48Ho8WLlyoO++885TWUVFRoauuumpU6hn+W3d2do7K+t6+TgAAAGCiIYgAAADAuHDrrbcqGAzq/vvvP+bzfX19euyxx3TVVVcpPz//tN9n0aJFWrt2rRYtWnTa6zgZ7xZEFBYWau3atbryyitT+v7H88wzz2jt2rV65pln9B//8R8qKyvTv/zLv2jOnDnatm3biGW/8Y1v6LHHHjut97nyyiu1du1aFRYWJh9LRRARj8d19dVX67XXXtN//Md/6JFHHtH111+vNWvWjOr7AAAAADg95nQXAAAAAEjSe9/7XhUVFenuu+/W5z//+aOeX716tQKBgG699dYzeh+v16tly5ad0TrOhM1mS+v7S9LixYuVk5OT/P3DH/6w/v7v/14XXXSR3v/+96u2tlY2m02SVFVVddrvk5ubq9zc3DOu90T27t2rrVu36q677tJHP/pRSdLll1+e8vcFAAAAcHIYEQEAAIBxwWQy6WMf+5g2bdqkHTt2HPX873//exUWFuq9732vOjo69PnPf16zZ8+W2+1WXl6eLr74Yr3++usnfJ93a810zz33aMaMGbLZbJo1a5b++Mc/HvP13/nOd7R06VJlZWXJ6/Vq0aJF+t3vfqdEIpFcpqKiQjU1NXr11VeTbZCG2xu9W2umN954Q5dccok8Ho+cTqdWrFihp5566qgaDQaDXn75ZX3uc59TTk6OsrOzdd1116m5ufmE23488+fP1x133KGGhgY9+OCDyceP1Zqpt7dXt956q7KysuR2u3XllVfq4MGDR7XBemdrppUrV+qpp55SfX39iBZRw+666y7Nnz9fbrdbHo9HM2fO1Ne//vUT1m4ymSQdCSRS7fnnn9c111yjkpIS2e12TZ06VZ/97GfftQVTY2OjrrvuOnm9Xvl8Pn3kIx9RR0fHUcs9+OCDWr58uVwul9xut6644gpt2bLlhPW89NJLWrlypbKzs+VwOFRWVqbrr79eQ0NDZ7ytAAAAwGghiAAAAMC48clPflIGg0F33333iMd37dql9evX62Mf+5hMJpO6u7slSd/61rf01FNP6fe//72mTJmilStXntTcD+90zz336BOf+IRmzZqlRx55RP/2b/+m7373u3rppZeOWraurk6f/exn9dBDD+nRRx/Vddddp3/4h3/Qd7/73eQyjz32mKZMmaKFCxdq7dq1Wrt27XHbG7366qu6+OKL1dfXp9/97ndavXq1PB6Prr766hGhwLBPfepTslgsuv/++/WTn/xEr7zyij7ykY+c8na/0/vf/35J0muvvfauywy3Qbr//vv1r//6r3rssce0dOlSrVq16oTrv/POO3XeeeepoKAg+XdZu3atJOmBBx7Q5z//eV100UV67LHH9Pjjj+uf/umf5Pf7T7je6dOna+XKlfrlL3+pxx9//OQ29jQdOHBAy5cv11133aXnnntO3/zmN/XWW2/p/PPPVyQSOWr5D3zgA5o6daoefvhhffvb39bjjz+uK664YsSyP/jBD3TjjTdq9uzZeuihh3TvvfdqYGBAF1xwgXbt2vWutdTV1enKK6+U1WrV3XffrWeeeUY/+tGP5HK5FA6HU7L9AAAAwOmgNRMAAADGjalTp+rCCy/Ufffdp5/85CeyWCySlAwmPvnJT0qSZsyYMWIi4lgspiuuuEJ1dXX6r//6L61cufKk3zMej+uOO+7QokWL9NhjjyXv0D///PM1bdo0FRUVjVj+97///YjXrly5UolEQr/4xS/0jW98QwaDQQsXLpTD4TjpNlBf/epXlZmZqVdeeUVut1uSdNVVV2nBggX653/+Z91www0jRg6sWrVK//Vf/5X8vbu7W//yL/+i1tZWFRQUnPS2v1N5ebkkHXd0xTPPPKM33nhDd911l2677TZJ0mWXXSar1aqvfe1rx13/7NmzlZGRccz2VGvWrFFGRsaI7brkkktOqu7a2lq1traqqqpKH/rQh/Too4+mbA6O4W2WpEQioRUrVmjlypUqLy/X008/nQxzhl133XX6yU9+IulIu6j8/HzdfPPNeuihh3TzzTersbFR3/rWt/T3f//3I7b9sssu07Rp0/Sd73znmGGUJG3atEnBYFA//elPNX/+/OTjN91002huMgAAAHDGGBEBAACAceXWW29VZ2ennnzySUlSNBrVfffdpwsuuEDTpk1LLverX/1KixYtkt1ul9lslsVi0Ysvvqjdu3ef0vvt3btXzc3Nuummm0Zc7C8vL9eKFSuOWv6ll17SpZdeKp/PJ5PJJIvFom9+85vq6upSe3v7KW+v3+/XW2+9pQ9+8IPJEEI60m7olltu0eHDh49qOfTOi93z5s2TJNXX15/y+7/d29tLvZtXX31VknTDDTeMePzGG288o/c+99xz1dvbqxtvvFFPPPHEu7Y6eqfu7m5deumluuyyy7Rjxw5dfvnluv766/X0008nl7nvvvtkMBh06NChM6pRktrb23XbbbeptLQ0ud8NBzjH2vduvvnmEb/fcMMNMpvNevnllyVJzz77rKLRqD760Y8qGo0mf+x2uy666KLjjvBZsGCBrFarPvOZz+gPf/iDDh48eMbbBwAAAKQCQQQAAADGlQ9+8IPy+XzJkQd/+ctf1NbWNmKS6p/97Gf63Oc+p6VLl+qRRx7RunXrtGHDBq1atUqBQOCU3q+rq0uSjjmS4J2PrV+/PjkJ8m9/+1utWbNGGzZs0B133CFJp/zektTT06NEIqHCwsKjnhsejTFc47Ds7OwRvw9PLH067/92w0HGO0eBvF1XV5fMZrOysrJGPJ6fn39G733LLbfo7rvvVn19va6//nrl5eVp6dKlev7554/7ut/97ndqbGzUN7/5TVmtVj3yyCO6/PLL9YEPfEDPPvuspCPzgsyaNUuVlZVnVGM8Htfll1+uRx99VP/yL/+iF198UevXr9e6deskHfvv/859yGw2Kzs7O/lv2tbWJkk655xzZLFYRvw8+OCDxw1kqqqq9MILLygvL09f+MIXVFVVpaqqKv3iF784o+0EAAAARhutmQAAADCuOBwO3Xjjjfrtb3+rlpYW3X333fJ4PPq7v/u75DL33XefVq5cqbvuumvEawcGBk75/YYv6re2th713Dsfe+CBB2SxWPTnP/9Zdrs9+fiZzEuQmZkpo9GolpaWo54bbpGUk5Nz2us/FcOjUI7X2io7O1vRaFTd3d0jwohj/f1O1Sc+8Ql94hOfkN/v12uvvaZvfetbuuqqq1RbW5scdfBOBw4ckMlkSo4msVqtevjhh/V3f/d3uvbaa/X//t//0x//+MejJgc/HTt37tS2bdt0zz336GMf+1jy8f3797/ra1pbW1VcXJz8PRqNqqurK7nfDf/bPvzww++6jcdzwQUX6IILLlAsFtPGjRv1y1/+Urfffrvy8/P14Q9/+JTXBwAAAKQCIyIAAAAw7tx6662KxWL66U9/qr/85S/68Ic/LKfTmXzeYDAkRwEM2759e3Li41MxY8YMFRYWavXq1SNaE9XX1+vNN98csazBYJDZbJbJZEo+FggEdO+99x61XpvNdlIjFFwul5YuXapHH310xPLxeFz33XefSkpKNH369FPerlO1bds2/eAHP1BFRcVRbZfe7qKLLpKko+YteOCBB07qfU7m7+JyufTe975Xd9xxh8LhsGpqat512blz5yoWi+l///d/k48NhxEXX3yxvvCFL2jFihWjMm/CcOuud+57v/71r9/1NW+vS5IeeughRaPRZNhzxRVXyGw268CBA1qyZMkxf06GyWTS0qVL9T//8z+SpM2bN5/sZgEAAAApx4gIAAAAjDtLlizRvHnz9POf/1yJRGJEWybpyETO3/3ud/Wtb31LF110kfbu3at///d/V2VlpaLR6Cm9l9Fo1He/+1196lOf0gc+8AF9+tOfVm9vr7797W8f1Vbnyiuv1M9+9jPddNNN+sxnPqOuri79x3/8x1EXpiWpurpaDzzwgB588EFNmTJFdrtd1dXVx6zhhz/8oS677DK95z3v0T//8z/LarXqzjvv1M6dO7V69eoRc1eMhk2bNsnn8ykSiai5uVkvvvii7r33XuXl5elPf/qTrFbru7521apVOu+88/TlL39Z/f39Wrx4sdauXas//vGPko78PY+nurpajz76qO666y4tXrxYRqNRS5Ys0ac//Wk5HA6dd955KiwsVGtrq374wx/K5/PpnHPOedf13Xrrrfr973+vz33uc9qxY4euuOIKxWIxrV27Vq+//rpKS0v1xhtv6KGHHjpuwDKstbVVDz/88FGPV1RUaP78+aqqqtJXv/pVJRIJZWVl6U9/+tNx20c9+uijMpvNuuyyy1RTU6NvfOMbmj9/frKWiooK/fu//7vuuOMOHTx4UKtWrVJmZqba2tq0fv16uVwufec73znmun/1q1/ppZde0pVXXqmysjIFg8HkxO6XXnrpCbcVAAAAGCsEEQAAABiXbr31Vn3xi1/U7NmztXTp0hHP3XHHHRoaGtLvfvc7/eQnP9Hs2bP1q1/9So899thxJ/c93ntJ0o9//GNdd911qqio0Ne//nW9+uqrI9Z38cUX6+6779aPf/xjXX311SouLtanP/1p5eXlHRWWfOc731FLS4s+/elPa2BgQOXl5aqrqzvm+1900UV66aWX9K1vfUsf//jHFY/HNX/+fD355JO66qqrTnl7TmTVqlWSjtzZn5WVperqav34xz/WJz7xCXk8nuO+1mg06k9/+pO+/OUv60c/+pHC4bDOO+883XfffVq2bJkyMjKO+/ovfvGLqqmp0de//nX19fUpkUgokUjoggsu0D333KOHHnpIPT09ysnJ0fnnn68//vGPys3Nfdf1ORwOvfbaa/rRj36khx56SHfeeaccDocWL16sX//617rhhhv0wQ9+UDfffLPMZrOuu+6649a3adOmEW3Ahn3sYx/TPffcoz/96U/64he/qM9+9rMym8269NJL9cILL6isrOyY63v00Uf17W9/W3fddZcMBoOuvvpq/fznPx8R9nzta1/T7Nmz9Ytf/EKrV69WKBRSQUGBzjnnHN12223vWuuCBQv03HPP6Vvf+pZaW1vldrs1d+5cPfnkk8m5TAAAAIDxwJB4+/hzAAAAADgN999/v26++WatWbNGK1asSHc5AAAAAMYRgggAAAAAp2T16tVqampSdXW1jEaj1q1bp5/+9KdauHChXn311XSXBwAAAGCcoTUTAAAAgFPi8Xj0wAMP6Hvf+578fr8KCwv18Y9/XN/73vfSXRoAAACAcYgREQAAAAAAAAAAIGWM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hjQVY4r9DmPtqquuUn5+vn73u98lH7v++uvldDp17733prEyTGbsdxgvsrKy9NOf/lS33nprukvBJBQIBOTxePTEE0/oyiuvTD6+YMECXXXVVfre976Xxuow2Q0ODmrRokW688479b3vfU8LFizQz3/+83SXhUmKz3ZIh6VLl2rRokW66667ko/NmjVL1157rX74wx+msTJMVhzrRjqV3IDWTCfw5JNPasmSJfq7v/s75eXlaeHChfrtb3+b7rIwybHfYaydf/75evHFF1VbWytJ2rZtm9544w29733vS3NlmMzY75BusVhMDzzwgPx+v5YvX57ucjBJRaNRxWIx2e32EY87HA698cYbaaoKZ4svfOELuvLKK3XppZemuxScBfhsh7EWDoe1adMmXX755SMev/zyy/Xmm2+mqSpMdhzrTh+tmU7g4MGDuuuuu/SlL31JX//617V+/Xr94z/+o2w2mz760Y+muzxMUux3GGv/+q//qr6+Ps2cOVMmk0mxWEzf//73deONN6a7NExi7HdIlx07dmj58uUKBoNyu9167LHHNHv27HSXhUnK4/Fo+fLl+u53v6tZs2YpPz9fq1ev1ltvvaVp06aluzxMYg888IA2b97MfCQYM3y2w1jr7OxULBZTfn7+iMfz8/PV2tqapqow2XGsO30EEScQj8e1ZMkS/eAHP5AkLVy4UDU1Nbrrrru4IIyUYb/DWHvwwQd133336f7779ecOXO0detW3X777SoqKtLHPvaxdJeHSYr9DukyY8YMbd26Vb29vXrkkUf0sY99TK+++iphBFLm3nvv1Sc/+UkVFxfLZDJp0aJFuummm7R58+Z0l4ZJqrGxUV/84hf13HPPHTUaB0gVPtshXQwGw4jfE4nEUY8Bo4Vj3ekjiDiBwsLCo76Uzpo1S4888kiaKsLZgP0OY+0rX/mKvvrVr+rDH/6wJKm6ulr19fX64Q9/yIkUKcN+h3SxWq3JyaqXLFmiDRs26Be/+IV+/etfp7kyTFZVVVV69dVX5ff71d/fr8LCQn3oQx9SZWVlukvDJLVp0ya1t7dr8eLFycdisZhee+01/fd//7dCoZBMJlMaK8RkxGc7jLWcnByZTKajRj+0t7cfNUoCGC0c604fc0ScwHnnnae9e/eOeKy2tlbl5eVpqghnA/Y7jLWhoSEZjSNPCSaTSfF4PE0V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",
      "text/plain": [
       "<Figure size 2000x400 with 11 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display_digits_with_boxes(validation_digits, validation_labels,\n",
    "                          validation_labels, np.array([]), validation_bboxes,\n",
    "                          np.array([]), \"Validation Digits & Labels\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c2f8803-9488-499f-9fed-600e473c851b",
   "metadata": {},
   "source": [
    "# 4. Define the Network"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "16ed4536-2b31-4eb5-8d3a-04aad5617173",
   "metadata": {},
   "outputs": [],
   "source": [
    " def feature_extractor(inputs):\n",
    "     x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3, input_shape=(75,75,1))(inputs)\n",
    "     x = tf.keras.layers.AveragePooling2D((2,2))(x)\n",
    "     \n",
    "     x = tf.keras.layers.Conv2D(32, activation='relu', kernel_size=3)(x)\n",
    "     x = tf.keras.layers.AveragePooling2D((2,2))(x)\n",
    "     \n",
    "     x = tf.keras.layers.Conv2D(64, activation='relu', kernel_size=3)(x)\n",
    "     x = tf.keras.layers.AveragePooling2D((2,2))(x)\n",
    "     \n",
    "     return x\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "862046e5-a389-4143-86dd-ed92d561c0ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "def dense_layers(inputs):\n",
    "    x = tf.keras.layers.Flatten()(inputs)\n",
    "    x= tf.keras.layers.Dense(128, activation='relu')(x)\n",
    "    return x    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "b13f203b-c5de-4dc3-8f94-e7c5d3bd8584",
   "metadata": {},
   "outputs": [],
   "source": [
    "def classifier(inputs):\n",
    "    classification_output = tf.keras.layers.Dense(10, activation=\"softmax\", name=\"classification\")(inputs)\n",
    "    return classification_output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "a7cef602-57d1-4625-a9f3-206dae1a7bdb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def bounding_box_regression(inputs ):\n",
    "    bounding_box_regression_output = tf.keras.layers.Dense(4, name = \"bounding_box\")(inputs)\n",
    "    return bounding_box_regression_output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "8be121eb-a870-4b5d-ba6c-fcbde75acee1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def final_model(inputs):\n",
    "    feature_cnn = feature_extractor(inputs)\n",
    "    dense_output = dense_layers(feature_cnn)\n",
    "\n",
    "    classification_output = classifier(dense_output)\n",
    "    bounding_box_output = bounding_box_regression(dense_output)\n",
    "\n",
    "    model = tf.keras.Model(inputs = inputs,outputs = [classification_output, bounding_box_output])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "29e453d1-89b1-4bd6-a25b-f1aa359d6a6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def define_and_compile_model(inputs):\n",
    "    model = final_model(inputs)\n",
    "    model.compile(optimizer = 'adam', loss = {'classification' : 'categorical_crossentropy',\n",
    "                                              'bounding_box' : 'mse'},\n",
    "                                      metrics = {'classification' : 'accuracy',\n",
    "                                                 'bounding_box' : 'mse'})\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "e84b33e3-687a-4f54-9880-14c9d79977a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\PARIMALA\\anaconda3\\Lib\\site-packages\\keras\\src\\layers\\convolutional\\base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"functional\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"functional\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)        </span>┃<span style=\"font-weight: bold\"> Output Shape      </span>┃<span style=\"font-weight: bold\">    Param # </span>┃<span style=\"font-weight: bold\"> Connected to      </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩\n",
       "│ input_layer         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">75</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">75</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>) │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ -                 │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">InputLayer</span>)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)     │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">73</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">73</span>,    │        <span style=\"color: #00af00; text-decoration-color: #00af00\">160</span> │ input_layer[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>] │\n",
       "│                     │ <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ average_pooling2d   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">36</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">36</span>,    │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ conv2d[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]      │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">AveragePooling2D</span>)  │ <span style=\"color: #00af00; text-decoration-color: #00af00\">16</span>)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">34</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">34</span>,    │      <span style=\"color: #00af00; text-decoration-color: #00af00\">4,640</span> │ average_pooling2… │\n",
       "│                     │ <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ average_pooling2d_1 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">17</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">17</span>,    │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ conv2d_1[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]    │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">AveragePooling2D</span>)  │ <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ conv2d_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">15</span>,    │     <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │ average_pooling2… │\n",
       "│                     │ <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ average_pooling2d_2 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)  │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ conv2d_2[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]    │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">AveragePooling2D</span>)  │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3136</span>)      │          <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │ average_pooling2… │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>)       │    <span style=\"color: #00af00; text-decoration-color: #00af00\">401,536</span> │ flatten[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]     │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ classification      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">10</span>)        │      <span style=\"color: #00af00; text-decoration-color: #00af00\">1,290</span> │ dense[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]       │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)             │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ bounding_box        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>)         │        <span style=\"color: #00af00; text-decoration-color: #00af00\">516</span> │ dense[<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>][<span style=\"color: #00af00; text-decoration-color: #00af00\">0</span>]       │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)             │                   │            │                   │\n",
       "└─────────────────────┴───────────────────┴────────────┴───────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)       \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape     \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m   Param #\u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mConnected to     \u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩\n",
       "│ input_layer         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m75\u001b[0m, \u001b[38;5;34m75\u001b[0m, \u001b[38;5;34m1\u001b[0m) │          \u001b[38;5;34m0\u001b[0m │ -                 │\n",
       "│ (\u001b[38;5;33mInputLayer\u001b[0m)        │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)     │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m73\u001b[0m, \u001b[38;5;34m73\u001b[0m,    │        \u001b[38;5;34m160\u001b[0m │ input_layer[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m] │\n",
       "│                     │ \u001b[38;5;34m16\u001b[0m)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ average_pooling2d   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m36\u001b[0m, \u001b[38;5;34m36\u001b[0m,    │          \u001b[38;5;34m0\u001b[0m │ conv2d[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]      │\n",
       "│ (\u001b[38;5;33mAveragePooling2D\u001b[0m)  │ \u001b[38;5;34m16\u001b[0m)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m34\u001b[0m, \u001b[38;5;34m34\u001b[0m,    │      \u001b[38;5;34m4,640\u001b[0m │ average_pooling2… │\n",
       "│                     │ \u001b[38;5;34m32\u001b[0m)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ average_pooling2d_1 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m17\u001b[0m, \u001b[38;5;34m17\u001b[0m,    │          \u001b[38;5;34m0\u001b[0m │ conv2d_1[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]    │\n",
       "│ (\u001b[38;5;33mAveragePooling2D\u001b[0m)  │ \u001b[38;5;34m32\u001b[0m)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ conv2d_2 (\u001b[38;5;33mConv2D\u001b[0m)   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m15\u001b[0m, \u001b[38;5;34m15\u001b[0m,    │     \u001b[38;5;34m18,496\u001b[0m │ average_pooling2… │\n",
       "│                     │ \u001b[38;5;34m64\u001b[0m)               │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ average_pooling2d_2 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m64\u001b[0m)  │          \u001b[38;5;34m0\u001b[0m │ conv2d_2[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]    │\n",
       "│ (\u001b[38;5;33mAveragePooling2D\u001b[0m)  │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m3136\u001b[0m)      │          \u001b[38;5;34m0\u001b[0m │ average_pooling2… │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m)       │    \u001b[38;5;34m401,536\u001b[0m │ flatten[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]     │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ classification      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m10\u001b[0m)        │      \u001b[38;5;34m1,290\u001b[0m │ dense[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]       │\n",
       "│ (\u001b[38;5;33mDense\u001b[0m)             │                   │            │                   │\n",
       "├─────────────────────┼───────────────────┼────────────┼───────────────────┤\n",
       "│ bounding_box        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m)         │        \u001b[38;5;34m516\u001b[0m │ dense[\u001b[38;5;34m0\u001b[0m][\u001b[38;5;34m0\u001b[0m]       │\n",
       "│ (\u001b[38;5;33mDense\u001b[0m)             │                   │            │                   │\n",
       "└─────────────────────┴───────────────────┴────────────┴───────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">426,638</span> (1.63 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m426,638\u001b[0m (1.63 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">426,638</span> (1.63 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m426,638\u001b[0m (1.63 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "with strategy.scope():\n",
    "    inputs = tf.keras.layers.Input(shape=(75,75,1,))\n",
    "    model = define_and_compile_model(inputs)\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d2300afd-1463-41e7-beb6-4853750eccd0",
   "metadata": {},
   "source": [
    "## 5.Train and Validate the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f06f42d2-a5d4-4555-a95b-a098e463f493",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "\u001b[1m278/937\u001b[0m \u001b[32m━━━━━\u001b[0m\u001b[37m━━━━━━━━━━━━━━━\u001b[0m \u001b[1m1:49\u001b[0m 166ms/step - bounding_box_loss: 0.0413 - bounding_box_mse: 0.0413 - classification_accuracy: 0.1918 - classification_loss: 2.1476 - loss: 2.1888"
     ]
    }
   ],
   "source": [
    "EPOCHS = 20\n",
    "steps_per_epoch = 60000 // BATCH_SIZE\n",
    "validation_steps = 1\n",
    "\n",
    "history = model.fit(training_dataset, steps_per_epoch = steps_per_epoch,\n",
    "                    validation_data = validation_dataset,validation_steps = 1, epochs=EPOCHS) \n",
    "\n",
    "loss, classification_loss, bounding_box_loss, classification_acc, bounding_box_mse = model.evaluate(validation_dataset, steps = 1)\n",
    "print(\"\\n-----------------------------------\\n\")\n",
    "print(\"Validation Accuracy: \" , classification_acc)\n",
    "print(\"\\n-----------------------------------\\n\")      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d39904e8-08b5-4a77-81e7-3dfcfaba50f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_metrics(\"bounding_box_mse\", \"Bounding Box MSE\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3720b5bb-858a-42ef-9da1-899ca35f21a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_metrics(\"classification_accuracy\", \"Classification Accuracy\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd9d705f-9ff2-4842-8ac7-d57f8b5d56f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_metrics(\"classification_loss\", \"Classification Loss\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e49883ee-aed4-45d2-9ea8-ce5bdaf98cbc",
   "metadata": {},
   "outputs": [],
   "source": [
    "def intersection_over_union(pred_box, true_box):\n",
    "    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis = 1)\n",
    "    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis = 1)\n",
    "\n",
    "    smoothing_factor = 1e-10\n",
    "\n",
    "    xmin_overlap = np.maximum(xmin_pred, xmin_true)\n",
    "    xmax_overlap = np.minimum(xmax_pred, xmax_true)\n",
    "    ymin_overlap = np.maximum(ymin_pred, ymin_true)\n",
    "    ymax_overlap = np.minimum(ymax_pred, ymax_true)\n",
    "\n",
    "    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)\n",
    "    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)  \n",
    "\n",
    "    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0) * np.maximum((ymax_overlap - ymin_overlap), 0)\n",
    "    union_area = (pred_box_area + true_box_area) - overlap_area\n",
    "\n",
    "    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)\n",
    "\n",
    "    return iou     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bb5d7b9-f562-4c02-b3a0-6833dd944c12",
   "metadata": {},
   "outputs": [],
   "source": [
    "prediction = model.predict(validation_digits, batch_size = 64)\n",
    "\n",
    "prediction_labels = np.argmax(prediction[0], axis = 1)\n",
    "\n",
    "prediction_bboxes = prediction[1]                       "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "40392b29-d210-4303-ad11-47255bf42573",
   "metadata": {},
   "outputs": [],
   "source": [
    "iou = intersection_over_union(prediction_bboxes, validation_bboxes)\n",
    "\n",
    "iou_threshold = 0.6\n",
    "\n",
    "display_digits_with_boxes(validation_digits, prediction_labels, validation_labels,\n",
    "                          prediction_bboxes, validation_bboxes, iou, \"True and Pred values\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9eb42820-2fba-43ab-bf25-a144d3033788",
   "metadata": {},
   "outputs": [],
   "source": [
    "display_digits_with_boxes(validation_digits, prediction_labels, validation_labels,\n",
    "                          prediction_bboxes, validation_bboxes, iou, \"True and Pred values\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5fc29177-bee2-4565-8cb3-b3b573693e8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "%history -f TensorflowProject.py"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
