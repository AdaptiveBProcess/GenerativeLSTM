import numpy as np
import pandas as pd
from operator import itemgetter
import pickle as pk
import os
import itertools

from support_modules.readers import log_reader as lr



def unpk(name):
    with open(name, 'rb') as fp:
        return pk.load(fp)

def dump_pk(obj,filename):
    with open(filename, 'wb') as fp:
        pk.dump(obj, fp)
    return

def create_all_prefixes(df):
    #Every prefix has start time, end time as meta features
    #as well as other attributes per task.
    # cpk.dump(all_seq, open('sequences.p', 'wb'))
    # cpk.dump(all_ids, open('ids.p', 'wb'))
    # cpk.dump(counts, open('counts.p', 'wb'))

    cur_id = df.at[0,'caseid']
    received_time = df.at[0,'start_time']
    prefix_db = []
    interval_db = []
    ts_db = []
    id_db = []
    #array of boolean values. True if complete prefix
    complete_p_db = []
    cur_prefix = []
    cur_ts = []
    #outcome_db = []
    for i in range(0,len(df)):
        print('Calculating '+str(i)+' out of '+str(len(df)))

        if cur_id!=df.at[i,'caseid']:

            complete_p_db[len(complete_p_db)-1] = True

            #end of sequence
            cur_id =df.at[i,'caseid']
            received_time = df.at[i, 'start_time']
            cur_prefix = []
            cur_prefix.append(df.at[i,'task'])
            cur_db = []
            cur_db.append(df.at[i,'start_time'])

            prefix_db.append(list(cur_prefix))
            cur_ts = []
            cur_ts.append(df.at[i, 'start_time'])

            ts_db.append(list(cur_ts))

            id_db.append(df.at[i,'caseid'])
#           ADAPTATION: the event id was included in the meta-data attributes
#           to enable the join with the original event log
#           interval_db.append([received_time,df.at[i,'start_time']])
            
            interval_db.append([received_time,df.at[i,'start_time'],df.at[i,'event_id']])
            complete_p_db.append(False)

        else:
            cur_prefix.append(df.at[i,'task'])
            cur_ts.append(df.at[i, 'start_time'])

            prefix_db.append(list(cur_prefix))
            ts_db.append(list(cur_ts))

            #Adding start and end time for the prefix
#            interval_db.append([received_time,df.at[i,'start_time']])
            interval_db.append([received_time,df.at[i,'start_time'],df.at[i,'event_id']])
            id_db.append(df.at[i,'caseid'])
            complete_p_db.append(False)
            #outcome_db.append(df.at[i,'outcome'])

    return [prefix_db,id_db,interval_db, ts_db,complete_p_db]#, outcome_db]


def return_longest_running(interval,id, prefixes, intervals,ids, complete,ts, comp_ind):
    #We need to find the longest prefix of a case that is still running (and the ts)
    #and its ID is not equal to id
    #ids = []

#    pref_id = []
    Q = []
#    ints = []
    for j in comp_ind:

        if ids[j]==id:
            continue
        else:
            #prefix is running if it started before the last time of the current prfeix
            #and ended after
            if intervals[j][1]>interval[1] and intervals[j][0]<interval[1]:
                Q.append([prefixes[j],ts[j]])
    return Q



def return_outcome(cur_time,cur_id,prefixes, ids, intervals, ts,complete):

    for j,p in enumerate(prefixes):
        if cur_id == ids[j] and complete[j] == True:
            #we found the end
            return intervals[j][1]-cur_time

    return 0



def is_recent_K(p,cur_time,q,K):


    ind = 0
    for j,t in enumerate(q):
        if t>cur_time:
            break
        else:
            ind = j
    #index contains the position of the most recent task - it
    #also indicates the length of the prefix we are interested in

    pref = q[0][0:ind+1]
#    ts = q[1][0:ind+1]

    cur = K

    while cur > 0:

        ind1 = len(pref)-(K-cur)-1
        ind2 = len(p) - (K-cur)-1
        if ind1<0 or ind2<0:
            break
        if pref[ind1] != p[ind2]:
            return False
        cur = cur-1


    return True


def return_recent_K(p, cur_time, Q, K):

    R = []

    for q in Q:
        #q is [0] being prefix, [1] being the timestamps series
        if is_recent_K(p,cur_time,q,K)==True:
            R.append(q)
    return R

def return_comp_ind(complete):
    indices = []
    for j,c in enumerate(complete):
        if c == True:
            indices.append(j)
    return indices

def metric1(i1,i2):

    #this is "city distance" metric
    return abs(i1[1]-i2[1]) + abs(i1[0]-i2[0])


def metric_snap(i1,i2):
    #snapshot:
    return i1[1]-i2[1]


def nearest_complete_prefix(cur_p,interval,prefixes, intervals,  ts, comp_ind, metric):
#    ind = 0
#
#    nearest_n = []
    comp_list = []
    for i,p in enumerate(prefixes):
        if i in comp_ind and ts[i][len(ts[i])-1]<interval[0]:
            if metric=='city':
                comp_list.append([intervals[i][1]-intervals[i][0],metric1(interval,intervals[i]), levenshteinDistance(cur_p,p)])
            elif metric=='snap':
                comp_list.append([intervals[i][1]-intervals[i][0],metric_snap(interval,intervals[i]),levenshteinDistance(cur_p,p)])
    return sorted(comp_list, key=itemgetter(1))



def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def return_longest_running_new(running_now, p, K):
    suff = []
    K = min(len(p),K)

    for i in range(1,K+1):
      suff.append(p[len(p)-i])

    running_now['suff_match'] = True

    for j in range(0,len(running_now)):
        flag = True
        for i in range(1,K+1):
            if len(running_now.at[j,'prefix'])<K:
                flag= False
                break
            if running_now.at[j,'prefix'][len(running_now.at[j,'prefix'])-i] != suff[i-1]:
                flag=False
                break
        running_now.at[j,'suff_match'] = flag

    running_now = running_now[running_now['suff_match']==True]
    Q = running_now.groupby(['id'], sort= False)['end_time'].max()

    return Q


def return_dde_city(running_now, start, end, threshold = 100):

    running_now['match'] = True

    for j in range(0,len(running_now)):

        metric = abs(running_now.at[j,'end_time']-end)+ \
                                        abs(running_now.at[j,'start_time']-start)

        flag = True
        if metric>threshold:
            flag = False
        running_now.at[j, 'match'] = flag


    running_now = running_now[running_now['match']==True]

    Q = running_now.groupby(['id'], sort= False)['end_time'].max()

    return Q

def return_dde_snap(running_now, start, end, threshold = 100):

    running_now.loc[:,'match'] = True

    for j in range(0,len(running_now)):

        metric = end - running_now.at[j,'end_time']

        flag = True
        if metric>threshold:
            flag = False
        running_now.at[j, 'match'] = flag


    running_now = running_now[running_now['match']==True]

    Q = running_now.groupby(['id'], sort= False)['end_time'].max()

    return Q
def return_outcome_new(end_time,id, df):

    df_temp = df[df.id == id]
    df_temp.reset_index(inplace=True, drop=True)
    if len(df_temp)==0:
        return -1
    return df_temp.at[0,'end_time']-end_time

def return_outcome_case(id, df):

    df_temp = df[df.id == id]
    df_temp.reset_index(inplace=True, drop=True)
    if len(df_temp)==0:
        return -1
    return df_temp.at[0,'outcome']

def return_running_now(start,end, id, df):
    running_now = df[(df.id != id) & (df.end_time < end) \
                     & (df.end_time > start)]
    running_now.reset_index(inplace=True, drop=True)
    return running_now

def feature_encoding_new(df):

    df_running = df[df.complete==False]
    df_complete = df[df.complete==True]

    df_running.reset_index(inplace=True, drop=True)

    df_complete.reset_index(inplace=True, drop=True)

    dataset = []

    #for every ts in ts / prefix in prefix, we create a feature vector x1...xp of size p and outcome y
    for j in range(0,len(df)):
        p = df.at[j,'prefix']
        # ADAPTATION: the outcome restriction was removed
#        print('Calculating '+str(j+1)+' out of '+str(len(df)))

#        outcome = return_outcome_new(df.at[j, 'end_time'],
#                                     df.at[j, 'id'], df_complete)
##        if outcome < 0:
##            continue

        feat = []
        #first feature, which we will (probably) not use in learning is id
        feat.append(df.at[j,'id'])
        
        # Extension added: event_id
        feat.append(df.at[j,'event_id'])
        
        #elapsed time, currently our only intra-case feature.
        feat.append((df.at[j,'end_time']- df.at[j,'start_time'])/60)

        #last task in prefix
        feat.append(p[len(p)-1])

        #L1,2,3 = Number of running prefixes
        run_now = return_running_now(df.at[j, 'start_time'],
                                     df.at[j, 'end_time'],
                                     df.at[j, 'id'],
                                     df=df)
        for k in [0,1,3]: #L1,2,3 in the paper
            feat.append(len(return_longest_running_new(run_now,
                                           p = df.at[j,'prefix'],
                                           K = k)))


        #for t in [600,1200,1800,2400,3600]:
        #City Distance features
        for t in [86400,432000,864000,1296000,2592000]:
            feat.append(len(return_dde_city(run_now,df.at[j, 'start_time'],
                                            df.at[j, 'end_time'],threshold = t)))
        #Chiara and Fabrizio
        #Snapshot distance features
        for t in [86400,432000,864000,1296000,2592000]:
        #for t in [600,1200,1800,2400,3600]:

            feat.append(len(return_dde_snap(run_now,df.at[j, 'start_time'],
                                            df.at[j, 'end_time'],threshold = t)))

#        feat.append(outcome)
        dataset.append(feat)

    return dataset

def extract_features(parms):
#    ADAPTATION: The output file and the auxiliar files routes were customised
    fname = os.path.splitext(parms['file_name'])[0]
    output_path = os.path.join('input_files', fname+'_inter')
    aux_files_path = os.path.join('support_modules', 'intercase_features')

#    ADAPTATION: The inclusion of any format of .csv event log and use single timestamps was added 
    df = read_log(parms)
   
    #Creating L*
    #prf[0] - all prefixes
    #prf[1] - ids of prefixes, so that we know who is who
    #prf[2] - meta data of prefix. Here it is start/end time.
    #pref[3] = timestamps
    #pref[4] - complete prefixes
    prf = create_all_prefixes(df)

    pk.dump(prf[0], open(os.path.join(aux_files_path, fname+'prefixes.p'), 'wb'))
    pk.dump(prf[1], open(os.path.join(aux_files_path, fname+'ids.p'), 'wb'))
    pk.dump(prf[2], open(os.path.join(aux_files_path, fname+'intervals.p'), 'wb'))
    pk.dump(prf[3], open(os.path.join(aux_files_path, fname+'ts.p'), 'wb'))
    pk.dump(prf[4], open(os.path.join(aux_files_path, fname+'complete.p'), 'wb'))
    #pk.dump(prf[5], open('outcome.p', 'wb'))
    
    
    prefixes = list(unpk(os.path.join(aux_files_path, fname+'prefixes.p')))
    ids = list(unpk(os.path.join(aux_files_path, fname+'ids.p')))
    intervals = list(unpk(os.path.join(aux_files_path, fname+'intervals.p')))
#    ts = list(unpk(os.path.join(aux_files_path, 'ts.p')))
    complete = list(unpk(os.path.join(aux_files_path, fname+'complete.p')))
    
    int_start = []
    int_end = []
    
    int_event_id = []
    
    for i in intervals:
        int_start.append(i[0])
        int_end.append(i[1])
        int_event_id.append(i[2])
    
    df_prefixes = pd.DataFrame({'id':ids, 'event_id':int_event_id, 'start_time':int_start,
                                'end_time': int_end,'complete':complete})#, 'outcome':outcomes})
    df_prefixes['prefix'] = ''
    
    
    for i,p in enumerate(prefixes):
        df_prefixes.at[i,'prefix'] = p    #','.join(p)
    
    dataset = feature_encoding_new(df_prefixes)
#     ADAPTATION: The outcome Y was removed since this is not used in our approach,
#     additionally the features of all prefixes were calculated 
    intercase_df = pd.DataFrame(dataset, columns=['id','event_id','elapsed','lasttask',
                                        'l1','l2','l3',
                                        'city1','city2','city3','city4','city5',
                                        'snap1','snap2','snap3','snap4','snap5'])

#    ADAPTATION: The merge with the original event log was included
    df = df.merge(intercase_df, on='event_id', how='left')
    df = df.drop(['end_time','start_time','event_id','ID','Elapsed','Lasttask'], axis=1)
    df.to_csv(output_path+'.csv', header = True)
 
    os.unlink(os.path.join(aux_files_path, fname+'prefixes.p'))
    os.unlink(os.path.join(aux_files_path, fname+'ids.p'))
    os.unlink(os.path.join(aux_files_path, fname+'intervals.p'))
    os.unlink(os.path.join(aux_files_path, fname+'ts.p'))
    os.unlink(os.path.join(aux_files_path, fname+'complete.p'))

    
# =============================================================================
# Adaptation methods
# =============================================================================

def read_log(parms):
    parms['read_options']['filter_d_attrib'] = True
    log = lr.LogReader(os.path.join('input_files', parms['file_name']), parms['read_options'])
    log_df = pd.DataFrame(log.data)
    log_df['end_time'] = log_df['end_timestamp'].astype(np.int64) // 10**9
    if parms['one_timestamp']:
        log_df = log_df.to_dict('records')
        log_df = sorted(log_df, key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('end_timestamp'))
            for i in range(0, len(events)):
                # In one-timestamp approach the first activity of the trace is taken as instant
                # since there is no previous timestamp to find a range
                if i == 0:
                    events[i]['start_time'] = events[i]['end_time'] - 1
                else:
                    events[i]['start_time'] = events[i - 1]['end_time']
        log_df = pd.DataFrame(log_df)
    else:
        print(log_df)
        log_df['start_time'] = log_df['start_timestamp'].astype(np.int64) // 10**9
    log_df = log_df[log_df.task != 'Start']
    log_df = log_df[log_df.task != 'End']
    log_df = log_df.reset_index(drop=True)
    log_df.sort_values(by=['caseid', 'start_time'], ascending=[True, True], inplace=True)
    log_df.reset_index(inplace=True, drop=True)
    log_df['event_id'] = log_df.index
    return log_df
