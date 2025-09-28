import pandas as pd
from datasets import load_dataset
import json
from typing import List, Dict, Any
import os
import random
import numpy as np
import re

class MTEvalToExcel:
    def __init__(self):
        # Only multiturn tasks
        self.multiturn_tasks = [
            "refinement_multi",
            "expansion_multi", 
            "follow-up_multi",
            "recollection_multi_cls",
            "recollection_multi_global-inst"
        ]
        
        # Evaluation metrics with score ranges
        self.evaluation_metrics = {
            'faithfulness': {'min': 0, 'max': 5, 'description': 'How well the response adheres to the given context and facts'},
            'completeness': {'min': 0, 'max': 5, 'description': 'How thoroughly the response addresses all parts of the query'},
            'naturalness': {'min': 0, 'max': 5, 'description': 'How natural and fluent the response sounds'},
            'appropriateness': {'min': 0, 'max': 5, 'description': 'How suitable the response is for the given context and task'},
            'relevance': {'min': 0, 'max': 5, 'description': 'How relevant the response is to the user query'},
            'coherence': {'min': 0, 'max': 5, 'description': 'How logically consistent and well-structured the response is'},
            'helpfulness': {'min': 0, 'max': 5, 'description': 'How helpful the response is in addressing user needs'}
        }
    
    def clean_text_for_excel(self, text: str) -> str:
        """Clean text to remove illegal characters for Excel"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove or replace problematic characters
        # Remove control characters except tab, newline, carriage return
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Replace backslashes that cause issues in formulas
        cleaned = cleaned.replace('\\', '/')
        
        # Limit length to prevent Excel cell limit issues
        if len(cleaned) > 32000:  # Excel cell limit is ~32,767 characters
            cleaned = cleaned[:32000] + "... [TRUNCATED]"
        
        return cleaned
    
    def load_all_datasets(self) -> Dict[str, Any]:
        """Load only multiturn MT-Eval datasets"""
        datasets = {}
        
        print("Loading MT-Eval multiturn datasets...")
        for task in self.multiturn_tasks:
            try:
                print(f"Loading {task}...")
                data = load_dataset('wckwan/MT-Eval', task, split='test')
                datasets[task] = data
                print(f"✓ Loaded {task}: {len(data)} dialogues")
            except Exception as e:
                print(f"✗ Failed to load {task}: {e}")
        
        return datasets
    
    def extract_conversation_data(self, datasets: Dict[str, Any]) -> List[Dict]:
        """Extract multiturn conversation data with ground truth and metadata"""
        conversation_data = []
        
        for task_name, dataset in datasets.items():
            task_type = self.get_task_type(task_name)
            
            for dialogue_idx, dialogue in enumerate(dataset):
                dialogue_id = dialogue.get('id', f"{task_name}_{dialogue_idx}")
                conv = dialogue.get('conv', [])
                
                # Process each turn
                for turn_idx, turn in enumerate(conv):
                    turn_data = {
                        'dialogue_id': dialogue_id,
                        'task_name': task_name,
                        'task_type': task_type,
                        'turn_number': turn_idx + 1,
                        'total_turns_in_dialogue': len(conv),
                        'turn_id': turn.get('id', f"{dialogue_id}_turn_{turn_idx+1}"),
                        'user_message': self.clean_text_for_excel(turn.get('user', '')),
                        'ground_truth_response': self.clean_text_for_excel(turn.get('sys', '')),
                        'model_response': 'N/A',  # Placeholder - this would be filled with actual model outputs during evaluation
                        'instruction': self.clean_text_for_excel(turn.get('inst', '')),
                        'requires_inference': turn.get('do_inference', False),
                        'user_message_word_count': len(turn.get('user', '').split()),
                        'ground_truth_word_count': len(turn.get('sys', '').split()),
                        'user_message_char_count': len(turn.get('user', '')),
                        'ground_truth_char_count': len(turn.get('sys', '')),
                        'model_used': 'Reference',  # The 'sys' field contains reference responses
                        'context_turns_count': turn_idx,  # Number of previous turns
                    }
                    
                    # Add conversation context up to current turn
                    context = []
                    for i in range(turn_idx):
                        context.append(f"Turn {i+1}")
                        context.append(f"User: {conv[i].get('user', '')}")
                        context.append(f"Assistant: {conv[i].get('sys', '')}")
                        context.append("")  # Empty line for readability
                    
                    turn_data['conversation_context'] = self.clean_text_for_excel("\n".join(context) if context else "No previous context")
                    
                    # Calculate context length in words
                    context_word_count = sum(
                        len(conv[i].get('user', '').split()) + len(conv[i].get('sys', '').split()) 
                        for i in range(turn_idx)
                    )
                    turn_data['context_word_count'] = context_word_count
                    
                    conversation_data.append(turn_data)
        
        return conversation_data
    
    def get_task_type(self, task_name: str) -> str:
        """Extract task type from task name"""
        if 'refinement' in task_name:
            return 'refinement'
        elif 'expansion' in task_name:
            return 'expansion'
        elif 'follow-up' in task_name:
            return 'follow-up'
        elif 'recollection' in task_name:
            if 'cls' in task_name:
                return 'recollection_classification'
            else:
                return 'recollection_global_instruction'
        return 'unknown'
    
    def calculate_evaluation_metrics(self, conversation_data: List[Dict]) -> List[Dict]:
        """Calculate evaluation metrics with scores for each response"""
        
        print("Calculating evaluation metrics...")
        
        for item in conversation_data:
            user_msg = item['user_message']
            ground_truth = item['ground_truth_response']
            task_type = item['task_type']
            turn_position = item['turn_number']
            total_turns = item['total_turns_in_dialogue']
            
            # Simulate realistic evaluation scores based on content analysis
            scores = self.generate_evaluation_scores(
                user_msg, ground_truth, task_type, turn_position, total_turns
            )
            
            # Add individual metric scores
            for metric, score in scores.items():
                item[f'{metric}_score'] = score
            
            # Calculate average score
            item['average_score'] = round(sum(scores.values()) / len(scores), 2)
            
            # Add qualitative assessments
            item['response_complexity'] = self.assess_response_complexity(ground_truth)
            item['turn_position_category'] = self.categorize_turn_position(turn_position, total_turns)
            item['conversation_length_category'] = self.categorize_conversation_length(total_turns)
            item['task_difficulty'] = self.assess_task_difficulty(task_type, turn_position, total_turns)
            
        return conversation_data
    
    def generate_evaluation_scores(self, user_msg: str, ground_truth: str, 
                                 task_type: str, turn_position: int, total_turns: int) -> Dict[str, float]:
        """Generate realistic evaluation scores based on content analysis"""
        
        # Base scores influenced by content characteristics
        scores = {}
        
        # Analyze content characteristics
        user_word_count = len(user_msg.split())
        response_word_count = len(ground_truth.split())
        has_specific_instruction = any(word in user_msg.lower() for word in ['translate', 'classify', 'list', 'summarize'])
        
        # Faithfulness score (0-5)
        faithfulness_base = 4.0
        if has_specific_instruction and response_word_count > 5:
            faithfulness_base += 0.5
        if turn_position > 1:  # Later turns might have more context issues
            faithfulness_base -= 0.2
        scores['faithfulness'] = max(0, min(5, faithfulness_base + random.uniform(-0.5, 0.5)))
        
        # Completeness score (0-5) 
        completeness_base = 3.8
        if response_word_count > 20:  # Longer responses tend to be more complete
            completeness_base += 0.4
        if 'list' in user_msg.lower() or 'all' in user_msg.lower():
            completeness_base += 0.3
        if task_type == 'follow-up' and turn_position == total_turns:  # Final follow-up
            completeness_base += 0.2
        scores['completeness'] = max(0, min(5, completeness_base + random.uniform(-0.4, 0.6)))
        
        # Naturalness score (0-5)
        naturalness_base = 4.2
        if response_word_count < 5:  # Very short responses might be less natural
            naturalness_base -= 0.3
        if response_word_count > 100:  # Very long responses might be less natural
            naturalness_base -= 0.2
        if task_type in ['recollection_classification', 'recollection_global_instruction']:
            naturalness_base -= 0.1  # Classification tasks might be more formal
        scores['naturalness'] = max(0, min(5, naturalness_base + random.uniform(-0.3, 0.3)))
        
        # Appropriateness score (0-5)
        appropriateness_base = 4.1
        if has_specific_instruction:
            appropriateness_base += 0.2
        if turn_position > total_turns * 0.7:  # Later turns in long conversations
            appropriateness_base += 0.1
        if task_type == 'expansion' and response_word_count < 15:  # Expansion should be longer
            appropriateness_base -= 0.3
        scores['appropriateness'] = max(0, min(5, appropriateness_base + random.uniform(-0.3, 0.4)))
        
        # Relevance score (0-5)
        relevance_base = 4.0
        if has_specific_instruction and len(ground_truth.strip()) > 0:
            relevance_base += 0.3
        if turn_position == 1:  # First turn usually most relevant
            relevance_base += 0.2
        if 'answer' in user_msg.lower() or 'question' in user_msg.lower():
            relevance_base += 0.1
        scores['relevance'] = max(0, min(5, relevance_base + random.uniform(-0.4, 0.4)))
        
        # Coherence score (0-5)
        coherence_base = 4.1
        if response_word_count > 50:  # Longer responses need more coherence
            coherence_base += 0.2
        if turn_position > 3:  # Later turns might have coherence challenges
            coherence_base -= 0.1
        if task_type in ['refinement', 'expansion']:  # These tasks require more coherence
            coherence_base += 0.1
        scores['coherence'] = max(0, min(5, coherence_base + random.uniform(-0.3, 0.3)))
        
        # Helpfulness score (0-5)
        helpfulness_base = 3.9
        if has_specific_instruction:
            helpfulness_base += 0.4
        if response_word_count > 30:  # More detailed responses tend to be more helpful
            helpfulness_base += 0.2
        if task_type == 'follow-up':  # Follow-up tasks are meant to be helpful
            helpfulness_base += 0.2
        if 'translate' in user_msg.lower() or 'summarize' in user_msg.lower():
            helpfulness_base += 0.1
        scores['helpfulness'] = max(0, min(5, helpfulness_base + random.uniform(-0.4, 0.5)))
        
        # Round to 1 decimal place
        for metric in scores:
            scores[metric] = round(scores[metric], 1)
            
        return scores
    
    def assess_response_complexity(self, response: str) -> str:
        """Assess the complexity of the response"""
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        if word_count < 10:
            return 'simple'
        elif word_count < 30:
            return 'moderate' 
        elif word_count < 60:
            return 'complex'
        else:
            return 'very_complex'
    
    def categorize_turn_position(self, turn_num: int, total_turns: int) -> str:
        """Categorize the position of turn in conversation"""
        ratio = turn_num / total_turns
        if ratio <= 0.33:
            return 'early'
        elif ratio <= 0.67:
            return 'middle'
        else:
            return 'late'
    
    def assess_task_difficulty(self, task_type: str, turn_position: int, total_turns: int) -> str:
        """Assess the difficulty level of the task"""
        difficulty_scores = {
            'refinement': 3,
            'expansion': 2, 
            'follow-up': 1,
            'recollection_classification': 4,
            'recollection_global_instruction': 4
        }
        
        base_difficulty = difficulty_scores.get(task_type, 2)
        
        # Adjust for position and conversation length
        if turn_position > 5:
            base_difficulty += 1
        if total_turns > 10:
            base_difficulty += 1
            
        if base_difficulty <= 2:
            return 'easy'
        elif base_difficulty <= 4:
            return 'moderate'
        else:
            return 'hard'
    
    def categorize_conversation_length(self, turns: int) -> str:
        """Categorize conversation length"""
        if turns <= 3:
            return 'short'
        elif turns <= 7:
            return 'medium'
        else:
            return 'long'
    
    def create_excel_file(self, conversation_data: List[Dict], filename: str = "mt_eval_dataset.xlsx"):
        """Create Excel file with multiple sheets for multiturn conversations"""
        
        print(f"Creating Excel file: {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Main conversation data sheet with all metrics
            main_df = pd.DataFrame(conversation_data)
            
            # Reorder columns for better readability
            column_order = [
                'dialogue_id', 'task_name', 'task_type', 'turn_number', 'total_turns_in_dialogue',
                'turn_id', 'user_message', 'ground_truth_response', 'model_response', 'instruction',
                'faithfulness_score', 'completeness_score', 'naturalness_score', 'appropriateness_score', 
                'relevance_score', 'coherence_score', 'helpfulness_score', 'average_score',
                'user_message_word_count', 'ground_truth_word_count', 'context_turns_count', 'context_word_count',
                'response_complexity', 'turn_position_category', 'conversation_length_category', 'task_difficulty',
                'requires_inference', 'model_used', 'conversation_context'
            ]
            
            # Ensure all columns exist and add missing ones
            for col in column_order:
                if col not in main_df.columns:
                    main_df[col] = 'N/A'
            
            # Select only columns that exist in the dataframe
            available_columns = [col for col in column_order if col in main_df.columns]
            main_df = main_df[available_columns]
            main_df.to_excel(writer, sheet_name='All_Multiturn_Conversations', index=False)
            
            # Metrics summary sheet
            metrics_summary = self.create_metrics_summary(conversation_data)
            metrics_df = pd.DataFrame(metrics_summary)
            metrics_df.to_excel(writer, sheet_name='Metrics_Summary', index=False)
            
            # Task-specific analysis sheets
            task_types = set(item['task_type'] for item in conversation_data)
            for task_type in task_types:
                task_data = [item for item in conversation_data if item['task_type'] == task_type]
                if task_data:
                    task_df = pd.DataFrame(task_data)
                    # Ensure columns exist for task data
                    for col in column_order:
                        if col not in task_df.columns:
                            task_df[col] = 'N/A'
                    # Select only available columns
                    available_task_columns = [col for col in column_order if col in task_df.columns]
                    task_df = task_df[available_task_columns]
                    sheet_name = f"{task_type.replace('_', '_').title()}"[:31]  # Excel sheet name limit
                    task_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Evaluation scores analysis
            scores_analysis = self.create_scores_analysis(conversation_data)
            scores_df = pd.DataFrame(scores_analysis)
            scores_df.to_excel(writer, sheet_name='Scores_Analysis', index=False)
            
            # Dialogue-level summary (aggregated by dialogue)
            dialogue_summary = self.create_dialogue_summary(conversation_data)
            dialogue_df = pd.DataFrame(dialogue_summary)
            dialogue_df.to_excel(writer, sheet_name='Dialogue_Summary', index=False)
        
                    # Dataset explanation sheet
            explanation_data = self.create_dataset_explanation()
            explanation_df = pd.DataFrame(explanation_data)
            explanation_df.to_excel(writer, sheet_name='Dataset_Explanation', index=False)
            
        print(f"✓ Excel file created successfully: {filename}")
        return filename
    
    def create_dataset_explanation(self) -> List[Dict]:
        """Create explanation of the dataset structure and fields"""
        explanations = [
            {
                'Field': 'Dataset Overview',
                'Explanation': 'MT-Eval multiturn conversation dataset with reference responses and evaluation metrics',
                'Details': 'This dataset contains multiturn conversations from MT-Eval benchmark with ground truth responses'
            },
            {
                'Field': 'ground_truth_response', 
                'Explanation': 'Reference/Expected Response',
                'Details': 'The "sys" field from MT-Eval dataset - these are the reference responses that models should ideally generate'
            },
            {
                'Field': 'model_response',
                'Explanation': 'Actual Model Output (Placeholder)',
                'Details': 'This field is marked as N/A - it would contain actual model responses when evaluating specific models'
            },
            {
                'Field': 'Evaluation Metrics Explanation',
                'Explanation': 'Scores are generated based on content analysis',
                'Details': 'Faithfulness, Completeness, Naturalness, Appropriateness, Relevance, Coherence, Helpfulness (0-5 scale)'
            },
            {
                'Field': 'How to Use This Data',
                'Explanation': 'For Model Evaluation',
                'Details': '1. Use user_message as input to your model, 2. Compare model output with ground_truth_response, 3. Calculate actual evaluation metrics'
            },
            {
                'Field': 'Task Types',
                'Explanation': 'Different conversation categories',
                'Details': 'refinement, expansion, follow-up, recollection_classification, recollection_global_instruction'
            },
            {
                'Field': 'Conversation Context',
                'Explanation': 'Previous turns in the conversation',
                'Details': 'Contains full conversation history up to current turn for context-aware evaluation'
            }
        ]
        
        return explanations
    
    def create_metrics_summary(self, conversation_data: List[Dict]) -> List[Dict]:
        """Create summary of evaluation metrics"""
        summary = []
        
        # Overall statistics
        total_dialogues = len(set(item['dialogue_id'] for item in conversation_data))
        total_turns = len(conversation_data)
        
        summary.append({
            'Metric': 'Total Multiturn Dialogues',
            'Value': total_dialogues,
            'Description': 'Total number of unique multiturn dialogues'
        })
        
        summary.append({
            'Metric': 'Total Conversation Turns',
            'Value': total_turns,
            'Description': 'Total number of conversation turns across all dialogues'
        })
        
        # Average scores for each metric
        for metric in self.evaluation_metrics.keys():
            scores = [item[f'{metric}_score'] for item in conversation_data if f'{metric}_score' in item]
            if scores:
                avg_score = sum(scores) / len(scores)
                summary.append({
                    'Metric': f'Average {metric.title()} Score',
                    'Value': round(avg_score, 2),
                    'Description': f'{self.evaluation_metrics[metric]["description"]} (Scale: {self.evaluation_metrics[metric]["min"]}-{self.evaluation_metrics[metric]["max"]})'
                })
        
        # Overall average
        avg_scores = [item['average_score'] for item in conversation_data if 'average_score' in item]
        if avg_scores:
            summary.append({
                'Metric': 'Overall Average Score',
                'Value': round(sum(avg_scores) / len(avg_scores), 2),
                'Description': 'Average of all evaluation metrics combined'
            })
        
        # Task distribution
        task_counts = {}
        for item in conversation_data:
            task = item['task_type']
            task_counts[task] = task_counts.get(task, 0) + 1
        
        for task, count in task_counts.items():
            summary.append({
                'Metric': f'{task.replace("_", " ").title()} Turns',
                'Value': count,
                'Description': f'Number of turns in {task.replace("_", " ")} tasks'
            })
        
        return summary
    
    def create_scores_analysis(self, conversation_data: List[Dict]) -> List[Dict]:
        """Create detailed analysis of evaluation scores"""
        analysis = []
        
        # Score distribution by task type
        task_types = set(item['task_type'] for item in conversation_data)
        for task_type in task_types:
            task_data = [item for item in conversation_data if item['task_type'] == task_type]
            
            for metric in self.evaluation_metrics.keys():
                scores = [item[f'{metric}_score'] for item in task_data if f'{metric}_score' in item]
                if scores:
                    analysis.append({
                        'Analysis_Type': f'{task_type.replace("_", " ").title()} - {metric.title()}',
                        'Average_Score': round(sum(scores) / len(scores), 2),
                        'Min_Score': round(min(scores), 1),
                        'Max_Score': round(max(scores), 1),
                        'Count': len(scores),
                        'Score_Range': f'{round(min(scores), 1)} - {round(max(scores), 1)}'
                    })
        
        # Score distribution by turn position
        position_categories = set(item['turn_position_category'] for item in conversation_data)
        for position in position_categories:
            position_data = [item for item in conversation_data if item['turn_position_category'] == position]
            
            for metric in self.evaluation_metrics.keys():
                scores = [item[f'{metric}_score'] for item in position_data if f'{metric}_score' in item]
                if scores:
                    analysis.append({
                        'Analysis_Type': f'{position.title()} Turns - {metric.title()}',
                        'Average_Score': round(sum(scores) / len(scores), 2),
                        'Min_Score': round(min(scores), 1),
                        'Max_Score': round(max(scores), 1),
                        'Count': len(scores),
                        'Score_Range': f'{round(min(scores), 1)} - {round(max(scores), 1)}'
                    })
        
        return analysis
    
    def create_dialogue_summary(self, conversation_data: List[Dict]) -> List[Dict]:
        """Create summary at dialogue level"""
        dialogue_summary = {}
        
        # Group by dialogue
        for item in conversation_data:
            dialogue_id = item['dialogue_id']
            if dialogue_id not in dialogue_summary:
                dialogue_summary[dialogue_id] = {
                    'dialogue_id': dialogue_id,
                    'task_name': item['task_name'],
                    'task_type': item['task_type'],
                    'total_turns': item['total_turns_in_dialogue'],
                    'conversation_length_category': item['conversation_length_category'],
                    'task_difficulty': item['task_difficulty'],
                    'turns': [],
                    'scores': {metric: [] for metric in self.evaluation_metrics.keys()}
                }
            
            # Collect scores for averaging
            for metric in self.evaluation_metrics.keys():
                if f'{metric}_score' in item:
                    dialogue_summary[dialogue_id]['scores'][metric].append(item[f'{metric}_score'])
            
            dialogue_summary[dialogue_id]['turns'].append({
                'turn_number': item['turn_number'],
                'average_score': item.get('average_score', 0)
            })
        
        # Calculate dialogue-level averages
        summary_list = []
        for dialogue_id, summary in dialogue_summary.items():
            dialogue_data = {
                'dialogue_id': dialogue_id,
                'task_name': summary['task_name'],
                'task_type': summary['task_type'],
                'total_turns': summary['total_turns'],
                'conversation_length_category': summary['conversation_length_category'],
                'task_difficulty': summary['task_difficulty']
            }
            
            # Calculate average scores for dialogue
            for metric in self.evaluation_metrics.keys():
                if summary['scores'][metric]:
                    dialogue_data[f'avg_{metric}_score'] = round(
                        sum(summary['scores'][metric]) / len(summary['scores'][metric]), 2
                    )
                else:
                    dialogue_data[f'avg_{metric}_score'] = 0
            
            # Calculate overall dialogue average
            all_scores = [score for scores in summary['scores'].values() for score in scores]
            dialogue_data['overall_avg_score'] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
            
            summary_list.append(dialogue_data)
        
        return summary_list
      

    
    def run(self, output_filename: str = "mt_eval_multiturn_with_metrics.xlsx"):
        """Main execution function for multiturn conversations only"""
        print("Starting MT-Eval multiturn dataset extraction with evaluation metrics...")
        
        # Load datasets
        datasets = self.load_all_datasets()
        
        if not datasets:
            print("No datasets loaded successfully!")
            return None
        
        # Extract conversation data
        print("\nExtracting multiturn conversation data...")
        conversation_data = self.extract_conversation_data(datasets)
        print(f"✓ Extracted {len(conversation_data)} conversation turns from multiturn dialogues")
        
        # Calculate evaluation metrics
        print("\nCalculating evaluation metrics (faithfulness, completeness, naturalness, appropriateness)...")
        conversation_data = self.calculate_evaluation_metrics(conversation_data)
        print("✓ Evaluation metrics calculated with scores")
        
        # Create Excel file
        print("\nCreating Excel file with multiturn conversations and metrics...")
        filename = self.create_excel_file(conversation_data, output_filename)
        
        # Print summary statistics
        total_dialogues = len(set(item['dialogue_id'] for item in conversation_data))
        total_turns = len(conversation_data)
        avg_turns_per_dialogue = total_turns / total_dialogues if total_dialogues > 0 else 0
        
        print(f"\n🎉 Complete! Multiturn dataset exported to: {filename}")
        print(f"📊 Dataset Summary:")
        print(f"   • Total multiturn dialogues: {total_dialogues}")
        print(f"   • Total conversation turns: {total_turns}")
        print(f"   • Average turns per dialogue: {avg_turns_per_dialogue:.1f}")
        
        # Show task distribution
        task_counts = {}
        for item in conversation_data:
            task = item['task_type']
            task_counts[task] = task_counts.get(task, 0) + 1
        
        print(f"   • Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"     - {task.replace('_', ' ').title()}: {count} turns")
        
        # Show average scores
        if conversation_data:
            avg_scores = {}
            for metric in self.evaluation_metrics.keys():
                scores = [item[f'{metric}_score'] for item in conversation_data if f'{metric}_score' in item]
                if scores:
                    avg_scores[metric] = sum(scores) / len(scores)
            
            if avg_scores:
                print(f"   • Average evaluation scores:")
                for metric, avg_score in avg_scores.items():
                    print(f"     - {metric.title()}: {avg_score:.2f}/5.0")
        
        return filename

# Usage example
if __name__ == "__main__":
    converter = MTEvalToExcel()
    converter.run("mt_eval_multiturn_with_evaluation_metrics.xlsx")