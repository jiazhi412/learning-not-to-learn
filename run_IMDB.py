from __future__ import print_function
import os
from itertools import product
from re import escape

if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('errors'):
    os.makedirs('errors')

# parameters
command_template = 'python main_IMDB.py --e {} --IMDB_train_mode {} --IMDB_test_mode {}'
# p1 = ['IMDB_LNL_paper']
p1 = ['IMDB_LNL_paper_111']
p2 = ['eb1', 'eb2']
p3 = ['eb1', 'eb2', 'unbiased']
# p3 = ['unbiased_test_balanced', 'unbiased_test_eb1', 'unbiased_test_eb2']
# p3 = ['unbiased_test_eb1', 'unbiased_test_eb2']

for p11, p22, p33 in product(p1, p2, p3):
    command = command_template.format(p11, p22, p33)
    job_name = f'{p11}-{p22}-{p33}'
    bash_file = '{}.sh'.format(job_name)
    with open( bash_file, 'w' ) as OUT:
        OUT.write('#!/bin/bash\n')
        OUT.write('#SBATCH --job-name={} \n'.format(job_name))
        OUT.write('#SBATCH --ntasks=1 \n')
        OUT.write('#SBATCH --account=other \n')
        OUT.write('#SBATCH --qos=premium \n')
        OUT.write('#SBATCH --partition=ALL \n')
        OUT.write('#SBATCH --cpus-per-task=4 \n')
        OUT.write('#SBATCH --gres=gpu:1 \n')
        OUT.write('#SBATCH --mem=16G \n')
        OUT.write('#SBATCH --time=5-00:00:00 \n')
        OUT.write('#SBATCH --exclude=vista18 \n')
        # OUT.write('#SBATCH --exclude=vista[06,07,10,11,13,17-20] \n')
        OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
        OUT.write('#SBATCH --error=errors/{}.out \n'.format(job_name))
        OUT.write('source ~/.bashrc\n')
        OUT.write('echo $HOSTNAME\n')
        OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
        OUT.write('conda activate pytorch\n')
        OUT.write(command)

    qsub_command = 'sbatch {}'.format(bash_file)
    os.system( qsub_command )
    os.system('rm -f {}'.format(bash_file))
    print( qsub_command )
    print( 'Submitted' )

# p2 = ['eb1_ex', 'eb2_ex']
# p3 = ['eb1_ex', 'eb2_ex', 'unbiased_ex']

# for p11, p22, p33 in product(p1, p2, p3):
#     command = command_template.format(p11, p22, p33)
#     job_name = f'{p11}-{p22}-{p33}'
#     bash_file = '{}.sh'.format(job_name)
#     with open( bash_file, 'w' ) as OUT:
#         OUT.write('#!/bin/bash\n')
#         OUT.write('#SBATCH --job-name={} \n'.format(job_name))
#         OUT.write('#SBATCH --ntasks=1 \n')
#         OUT.write('#SBATCH --account=other \n')
#         OUT.write('#SBATCH --qos=premium \n')
#         OUT.write('#SBATCH --partition=ALL \n')
#         OUT.write('#SBATCH --cpus-per-task=4 \n')
#         OUT.write('#SBATCH --gres=gpu:1 \n')
#         OUT.write('#SBATCH --mem=16G \n')
#         OUT.write('#SBATCH --time=5-00:00:00 \n')
#         OUT.write('#SBATCH --exclude=vista18 \n')
#         # OUT.write('#SBATCH --exclude=vista[06,07,10,11,13,17-20] \n')
#         OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
#         OUT.write('#SBATCH --error=errors/{}.out \n'.format(job_name))
#         OUT.write('source ~/.bashrc\n')
#         OUT.write('echo $HOSTNAME\n')
#         OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
#         OUT.write('conda activate pytorch\n')
#         OUT.write(command)

#     qsub_command = 'sbatch {}'.format(bash_file)
#     os.system( qsub_command )
#     os.system('rm -f {}'.format(bash_file))
#     print( qsub_command )
#     print( 'Submitted' )
