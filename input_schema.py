INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["movie still, ocean, island, (clouds:0.4), epic, film grain"]
    },
    "negative_prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["(octane render, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, ugly face, bad face), (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured"]
    },
    "height": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [1024]
    },
    "width": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [1024]
    },
    "steps": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [5]
    },
    "guidance_scale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [7.5]
    }
    
}
