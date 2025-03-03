# interactive-seg-gui
MVP GUI for tkinter interactive segmentation app: data viewing, labelling, segementation display 

Components:

- GUI:
    - zoomable canvas (no patching!)
    - brush only (and eraser)
    
- Data model:
    - stores images, labels, feature stacks
    - caching (or is that handled by backend?)
    - classifiers live on separate thread (not data model)
    - labelling:
        - store diffs?
        - 

