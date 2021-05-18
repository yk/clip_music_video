from textwrap3          import fill
import moviepy.editor   as me
import tempfile
import textwrap
import glob
import os



def createvid(description, image_temp_list, fps=24, duration=0.1):
    blackbg = me.ColorClip((720,720), (0, 0, 0))

    clips = [me.ImageClip(m.name+".png", duration=duration) for m in image_temp_list]
    for img in image_temp_list:
        img.close()
    concat_clip = me.concatenate_videoclips(clips, method="compose").set_position(('center', 'center'))
    if description == "start song":
        description = " "
    if len(description) > 35:
        description = fill(description, 35)

    txtClip = me.TextClip(description, color='white', fontsize=30, font='Amiri-regular').set_position('center')
    txt_col = txtClip.on_color(size=(blackbg.w, txtClip.h + 10),
                               color=(0,0,0), pos=('center', 'center'), 
                               col_opacity=0.8)

    txt_mov = txt_col.set_position((0, blackbg.h-20-txtClip.h))
    comp_list = [blackbg, concat_clip, txt_mov]
    final = me.CompositeVideoClip(comp_list).set_duration(concat_clip.duration)

    with tempfile.NamedTemporaryFile() as video_tempfile:
        final.write_videofile(video_tempfile.name+".mp4", fps=fps)
        video_tempfile.seek(0)

        for clip in clips:
            clip.close()
        for clip in comp_list:
            clip.close()
        return video_tempfile

def concatvids(descriptions, video_temp_list, audiofilepath, fps=24, lyrics=True):
    clips = []

    for idx, (desc, vid) in enumerate(zip(descriptions, video_temp_list)):
    #     # compvid_list = []
    #     # desc = desc[1]
        if desc == descriptions[-1][1]:
            break
    #     # elif desc == "start song":
    #     #     desc = " "
    #     # compvid_list.append(blackbg)
        vid = me.VideoFileClip(f'{vid.name}.mp4')#.set_position(('center', 'center'))
        # compvid_list.append(vid)

        # if len(desc) > 35:
        #     desc = fill(desc, 35)
        # if lyrics:

        #     txtClip = me.TextClip(desc, color='white', fontsize=30, font='Amiri-regular').set_position('center')
        #     txt_col = txtClip.on_color(size=(blackbg.w, txtClip.h + 10),
        #               color=(0,0,0), pos=('center', 'center'), col_opacity=0.8)
            
        #     txt_mov = txt_col.set_position((0, blackbg.h-20-txtClip.h))
        #     compvid_list.append(txt_mov)

        # video_tempfile = tempfile.NamedTemporaryFile()
        
        # final = me.CompositeVideoClip(compvid_list).set_duration(vid.duration)
        # final.write_videofile(video_tempfile.name+".mp4", fps=fps)
        # video_tempfile.seek(0)
        # for clip in clips:
        #     clip.close()
        # concat_clip.close()


        clips.append(vid)

    concat_clip = me.concatenate_videoclips(clips, method="compose").set_position(('center', 'center'))
    # concat_clip = me.CompositeVideoClip([blackbg, concat_clip])#.set_duration(vid.duration)
    if audiofilepath:
        concat_clip.audio = me.AudioFileClip(audiofilepath)

        concat_clip.duration = concat_clip.audio.duration
    concat_clip.write_videofile(os.path.join('output', f"finaloutput.mp4"), fps=fps)



