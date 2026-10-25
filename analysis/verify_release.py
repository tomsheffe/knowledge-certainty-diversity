"""Check release populations, joins and the core PCS results without model calls."""
from pathlib import Path
import numpy as np
import pandas as pd
from common import load_exp4_human_llm, load_bot_bot_duo, pair_diversity_gain
from exp4_selection_table import load as selection_population

ROOT=Path(__file__).resolve().parents[1]


def main():
    full=pd.read_csv(ROOT/'data/exp4_human_llm_responses.csv')
    post=pd.read_csv(ROOT/'data/exp4_human_llm_discussions.csv')
    transcripts=pd.read_csv(ROOT/'data/conversations/experiment4.csv')
    assert len(full)==1550 and full.participant.nunique()==31
    assert 'group' not in full and 'doc.group' not in post
    assert len(post)==562 and len(load_exp4_human_llm())==561
    assert not full.duplicated(['participant','question_number']).any()
    assert post.release_participant.notna().all()
    assert len(transcripts)==len(full)
    merged=full.merge(transcripts,on=['participant','question_number'],validate='one_to_one')
    assert len(merged)==1550
    assert set(post.release_participant).issubset(set(full.participant))
    assert int(full.conversation_empty.sum())==214
    assert selection_population().grp.value_counts().to_dict()=={
        'discussed':562,'LLM answer shown':774,'LLM never spoke':214}
    d,_=load_bot_bot_duo()
    assert np.isclose(pair_diversity_gain(d)[0]*100,0.0,atol=1e-10)
    assert np.isclose(pair_diversity_gain(load_exp4_human_llm())[0]*100,8.0,atol=.1)
    trio = pd.read_csv(ROOT/'data/conversations/experiment3.csv')
    assert len(trio) == 410 and int(trio.privacy_redacted.sum()) == 41
    assert int(transcripts.privacy_redacted.sum()) == 3
    for frame in [trio, transcripts]:
        marked = frame.conversation.str.contains(r"\[(?:name|personal information|email|phone) redacted\]", na=False)
        assert frame.privacy_redacted.eq(marked).all()
    assert int(transcripts.source_text_incomplete.sum()) == 10
    assert not trio.duplicated(['group', 'question_number']).any()
    assert set(trio.question_number) == set(range(1, 43)) - {25}
    assert trio.groupby('group').size().eq(41).all()
    human = pd.read_csv(ROOT/'data/exp3_human_trios.csv')
    assert set(trio['group']).issubset(set(human.group_num))
    groups = pd.read_csv(ROOT/'data/aggregated/group_level_pcs_vs_gain.csv')
    assert len(groups) == 56 and groups[['pcs', 'gain']].notna().all().all()
    for path in (ROOT/'data').rglob('*.csv'):

        cols=set(pd.read_csv(path,nrows=0).columns)
        assert not cols.intersection({'phone_number','name_heb','chat_id','doc_id','age','gender','university','study_year'}),path
    for folder in ['data', 'derived_data']:
        for path in (ROOT/folder).rglob('*.csv'):
            frame = pd.read_csv(path)
            if 'Correct_answer' not in frame:
                continue
            assert {'item_id', 'upstream_id', 'Speaker_Order'}.issubset(frame.columns), path
            assert frame.item_id.str.fullmatch(r'[0-9a-f]{64}').all(), path
            for order in frame.Speaker_Order.dropna():
                assert all(n.strip() in {'Alice', 'David', 'John'} for n in order.split(',')), path
    print('PASS: populations, transcript links, required fields, and core PCS estimates.')


if __name__=='__main__':main()
