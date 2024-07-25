
"""
script to calculate planned time for data collection
"""

## static parameters
video_number = 1950
avg_video_time = 7
native_fps = 60
second2hour = 3600

### Changeable Parameters
avg_fps = 21
processing_ratio = 60 / avg_fps

### calculation
total_time = video_number * avg_video_time * processing_ratio / second2hour
print(f"Total hours needed to process {video_number} video data: {round(total_time, 2)} hrs")